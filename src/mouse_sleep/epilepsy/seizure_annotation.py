"""Channel-blind decoding and clinical evaluation of continuous seizures.

The decoder consumes only a one-dimensional probability trace and window
timestamps.  It deliberately has no animal- or channel-specific arguments, so
one validation-fitted configuration can be frozen and reused for every valid
single-channel recording.

Intervals are half-open ``[start_ms, end_ms)``.  A decision timestamp is the
centre of its four-second model window; the decoded offset is one nominal
stride after the final positive centre.  Boundary refinement operates on
changes between adjacent window-centre logits.  The resulting boundary still
has an intrinsic uncertainty of half the model-window duration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import ceil, floor, isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, order=True)
class EventInterval:
    """A half-open seizure interval in milliseconds."""

    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        if not isfinite(self.start_ms) or not isfinite(self.end_ms):
            raise ValueError("event boundaries must be finite")
        if self.end_ms <= self.start_ms:
            raise ValueError("event end_ms must be greater than start_ms")

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class DecoderConfig:
    """One global operating point fitted on validation recordings only."""

    high_threshold: float = 0.70
    low_threshold: float = 0.40
    median_width_windows: int = 3
    persistence_k: int = 2
    persistence_n: int = 3
    max_gap_seconds: float = 2.0
    min_event_seconds: float = 3.0
    boundary_search_seconds: float = 4.0
    window_seconds: float = 4.0
    stride_seconds: float = 1.0
    probability_clip: float = 1e-6
    continuity_gap_factor: float = 1.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_threshold < self.high_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= low_threshold < "
                "high_threshold <= 1"
            )
        if self.median_width_windows < 1 or self.median_width_windows % 2 == 0:
            raise ValueError("median_width_windows must be a positive odd integer")
        if not 1 <= self.persistence_k <= self.persistence_n:
            raise ValueError("persistence must satisfy 1 <= k <= n")
        if self.max_gap_seconds < 0.0:
            raise ValueError("max_gap_seconds cannot be negative")
        if self.min_event_seconds <= 0.0:
            raise ValueError("min_event_seconds must be positive")
        if self.boundary_search_seconds < 0.0:
            raise ValueError("boundary_search_seconds cannot be negative")
        if self.window_seconds <= 0.0 or self.stride_seconds <= 0.0:
            raise ValueError("window_seconds and stride_seconds must be positive")
        if not 0.0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must lie strictly between 0 and 0.5")
        if self.continuity_gap_factor <= 1.0:
            raise ValueError("continuity_gap_factor must be greater than 1")

    @property
    def boundary_uncertainty_seconds(self) -> float:
        """Uncertainty imposed by a score computed from a finite input window."""

        return self.window_seconds / 2.0


@dataclass(frozen=True)
class DecodedAnnotation:
    """Trace-level outputs needed for review figures and interval analysis."""

    config: DecoderConfig
    window_start_time_ms: np.ndarray
    window_center_time_ms: np.ndarray
    probability: np.ndarray
    smoothed_probability: np.ndarray
    smoothed_logit: np.ndarray
    seizure_mask: np.ndarray
    coarse_events: tuple[EventInterval, ...]
    events: tuple[EventInterval, ...]

    @property
    def boundary_uncertainty_seconds(self) -> float:
        return self.config.boundary_uncertainty_seconds


@dataclass(frozen=True)
class WindowIntervalDecodedAnnotation:
    """Interval-aware decoding of finite-window probabilities.

    The original decoder represents a positive decision by a one-stride
    interval at the window centre.  That is appropriate for centre-labelled
    classification, but it under-represents the temporal support of a score
    computed from a four-second waveform.  This result keeps the same
    hysteresis mask while using the physical ``[window_start, window_end)``
    interval for event coverage and boundary review.
    """

    config: DecoderConfig
    window_start_time_ms: np.ndarray
    window_end_time_ms: np.ndarray
    window_center_time_ms: np.ndarray
    probability: np.ndarray
    smoothed_probability: np.ndarray
    smoothed_logit: np.ndarray
    seizure_mask: np.ndarray
    coarse_events: tuple[EventInterval, ...]
    events: tuple[EventInterval, ...]

    @property
    def boundary_uncertainty_seconds(self) -> float:
        return self.config.boundary_uncertainty_seconds


@dataclass(frozen=True)
class EventMatch:
    """One member of the maximum-total-IoU one-to-one assignment."""

    true_index: int
    predicted_index: int
    overlap_ms: float
    iou: float
    dice: float
    annotated_coverage: float
    onset_error_seconds: float
    offset_error_seconds: float
    duration_error_seconds: float
    duration_ratio: float


@dataclass(frozen=True)
class TrueEventAssessment:
    """Coverage and fragmentation for one annotated event."""

    true_index: int
    annotated_duration_ms: float
    covered_duration_ms: float
    coverage: float
    overlapping_predicted_segments: int


@dataclass(frozen=True)
class EventEvaluation:
    metrics: dict[str, float | int]
    matches: tuple[EventMatch, ...]
    true_events: tuple[EventInterval, ...]
    predicted_events: tuple[EventInterval, ...]
    true_event_assessments: tuple[TrueEventAssessment, ...]


@dataclass(frozen=True)
class ValidationSequence:
    """Identity-free validation input for fitting a shared decoder."""

    probabilities: np.ndarray
    window_start_time_ms: np.ndarray
    true_events: tuple[EventInterval, ...]
    recording_duration_seconds: float

    def __post_init__(self) -> None:
        probabilities = _one_dimensional_float(
            self.probabilities, "probabilities", allow_empty=True
        ).copy()
        starts = _one_dimensional_float(
            self.window_start_time_ms, "window_start_time_ms", allow_empty=True
        ).copy()
        if len(probabilities) != len(starts):
            raise ValueError("probabilities and window_start_time_ms must align")
        if self.recording_duration_seconds <= 0.0 or not isfinite(
            self.recording_duration_seconds
        ):
            raise ValueError("recording_duration_seconds must be finite and positive")
        probabilities.setflags(write=False)
        starts.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "window_start_time_ms", starts)
        object.__setattr__(self, "true_events", _normalise_intervals(self.true_events))


@dataclass(frozen=True)
class DecoderCandidateResult:
    config: DecoderConfig
    score: float
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class DecoderFitResult:
    """Auditable result of validation-only global decoder selection."""

    config: DecoderConfig
    objective: str
    score: float
    metrics: dict[str, float | int]
    candidates: tuple[DecoderCandidateResult, ...]


def rolling_median(values: Sequence[float] | np.ndarray, width: int) -> np.ndarray:
    """Return an edge-padded centred rolling median."""

    array = _one_dimensional_float(values, "values", allow_empty=True)
    if width < 1 or width % 2 == 0:
        raise ValueError("width must be a positive odd integer")
    if not len(array) or width == 1:
        return array.copy()
    half = width // 2
    padded = np.pad(array, (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=-1)


def hysteresis_k_of_n(
    probabilities: Sequence[float] | np.ndarray,
    high_threshold: float,
    low_threshold: float,
    k: int,
    n: int,
) -> np.ndarray:
    """Decode a trace with backdated k-of-n onset and offset confirmation.

    An event begins once at least ``k`` of the latest ``n`` scores meet the
    high threshold.  It remains active until at least ``k`` of the latest
    ``n`` scores fall below the low threshold.  Confirmation is backdated to
    the first supporting sample, which avoids adding an artificial detection
    delay to retrospective boundary estimates.
    """

    values = _one_dimensional_float(probabilities, "probabilities", allow_empty=True)
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
    if not 1 <= k <= n:
        raise ValueError("persistence must satisfy 1 <= k <= n")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    active = False
    mask = np.zeros(len(values), dtype=bool)
    for index in range(len(values)):
        left = max(0, index - n + 1)
        recent = values[left : index + 1]
        if not active:
            high_indices = np.flatnonzero(recent >= high_threshold)
            if len(high_indices) >= k:
                onset = left + int(high_indices[-k])
                mask[onset : index + 1] = True
                active = True
            continue

        mask[index] = True
        low_indices = np.flatnonzero(recent < low_threshold)
        if len(low_indices) >= k:
            offset = left + int(low_indices[-k])
            mask[offset : index + 1] = False
            active = False
    return mask


def fill_short_negative_gaps(mask: Sequence[bool] | np.ndarray, max_gap: int) -> np.ndarray:
    """Fill interior False runs containing at most ``max_gap`` decisions."""

    output = _one_dimensional_bool(mask, "mask").copy()
    if max_gap < 0:
        raise ValueError("max_gap cannot be negative")
    if max_gap == 0:
        return output
    index = 0
    while index < len(output):
        if output[index]:
            index += 1
            continue
        end = index + 1
        while end < len(output) and not output[end]:
            end += 1
        if index > 0 and end < len(output) and end - index <= max_gap:
            output[index:end] = True
        index = end
    return output


def remove_short_positive_runs(
    mask: Sequence[bool] | np.ndarray, minimum_run: int
) -> np.ndarray:
    """Remove positive runs shorter than ``minimum_run`` decisions."""

    output = _one_dimensional_bool(mask, "mask").copy()
    if minimum_run < 1:
        raise ValueError("minimum_run must be positive")
    index = 0
    while index < len(output):
        if not output[index]:
            index += 1
            continue
        end = index + 1
        while end < len(output) and output[end]:
            end += 1
        if end - index < minimum_run:
            output[index:end] = False
        index = end
    return output


def decode_window_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    window_start_time_ms: Sequence[float] | np.ndarray,
    config: DecoderConfig,
) -> DecodedAnnotation:
    """Decode one continuous single-channel probability trace into events."""

    probability = _one_dimensional_float(
        probabilities, "probabilities", allow_empty=True
    ).copy()
    starts = _one_dimensional_float(
        window_start_time_ms, "window_start_time_ms", allow_empty=True
    ).copy()
    if len(probability) != len(starts):
        raise ValueError("probabilities and window_start_time_ms must align")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if len(starts) > 1 and np.any(np.diff(starts) <= 0.0):
        raise ValueError("window_start_time_ms must be strictly increasing")

    centres = starts + config.window_seconds * 500.0
    smoothed = np.empty_like(probability)
    smoothed_logit = np.empty_like(probability)
    seizure_mask = np.zeros(len(probability), dtype=bool)
    coarse_events: list[EventInterval] = []
    refined_events: list[EventInterval] = []
    stride_ms = config.stride_seconds * 1000.0
    max_gap_steps = floor(config.max_gap_seconds / config.stride_seconds + 1e-12)
    minimum_run_steps = ceil(
        config.min_event_seconds / config.stride_seconds - 1e-12
    )
    search_steps = floor(
        config.boundary_search_seconds / config.stride_seconds + 1e-12
    )

    for segment in _contiguous_slices(
        centres, stride_ms * config.continuity_gap_factor
    ):
        segment_probability = probability[segment]
        segment_smoothed = rolling_median(
            segment_probability, config.median_width_windows
        )
        segment_clipped = np.clip(
            segment_smoothed,
            config.probability_clip,
            1.0 - config.probability_clip,
        )
        segment_logit = np.log(segment_clipped / (1.0 - segment_clipped))
        segment_mask = hysteresis_k_of_n(
            segment_smoothed,
            config.high_threshold,
            config.low_threshold,
            config.persistence_k,
            config.persistence_n,
        )
        segment_mask = fill_short_negative_gaps(segment_mask, max_gap_steps)
        segment_mask = remove_short_positive_runs(segment_mask, minimum_run_steps)

        smoothed[segment] = segment_smoothed
        smoothed_logit[segment] = segment_logit
        seizure_mask[segment] = segment_mask
        segment_centres = centres[segment]
        segment_coarse = _events_from_mask(
            segment_centres, segment_mask, stride_ms
        )
        segment_refined = _refine_boundaries(
            segment_centres,
            segment_logit,
            segment_mask,
            stride_ms,
            search_steps,
        )
        coarse_events.extend(segment_coarse)
        refined_events.extend(segment_refined)

    for array in (starts, centres, probability, smoothed, smoothed_logit, seizure_mask):
        array.setflags(write=False)
    return DecodedAnnotation(
        config=config,
        window_start_time_ms=starts,
        window_center_time_ms=centres,
        probability=probability,
        smoothed_probability=smoothed,
        smoothed_logit=smoothed_logit,
        seizure_mask=seizure_mask,
        coarse_events=tuple(coarse_events),
        events=tuple(refined_events),
    )


def decode_window_intervals(
    probabilities: Sequence[float] | np.ndarray,
    window_start_time_ms: Sequence[float] | np.ndarray,
    config: DecoderConfig,
    window_end_time_ms: Sequence[float] | np.ndarray | None = None,
) -> WindowIntervalDecodedAnnotation:
    """Decode probabilities while preserving the support of each EEG window.

    A model score is computed from a finite waveform window, not an
    instantaneous sample.  The legacy decoder converted each positive centre
    into a one-stride interval, which can make a correctly detected event look
    like a short overlap and can shift the displayed onset/offset by roughly
    half a window.  This decoder applies the identical smoothing, hysteresis,
    gap filling and minimum-run rules, then takes the union of the physical
    window intervals for each positive run.  No labels are used here.

    ``window_end_time_ms`` is optional for compatibility; when omitted it is
    derived from ``config.window_seconds``.  Intervals are half-open and are
    merged only when their physical gap is no larger than
    ``config.max_gap_seconds``.  The returned ``events`` are therefore the
    intervals used by the corrected clinical audit, while ``seizure_mask``
    remains a per-window decision trace for plotting.
    """

    probability = _one_dimensional_float(
        probabilities, "probabilities", allow_empty=True
    ).copy()
    starts = _one_dimensional_float(
        window_start_time_ms, "window_start_time_ms", allow_empty=True
    ).copy()
    if len(probability) != len(starts):
        raise ValueError("probabilities and window_start_time_ms must align")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if len(starts) > 1 and np.any(np.diff(starts) <= 0.0):
        raise ValueError("window_start_time_ms must be strictly increasing")

    if window_end_time_ms is None:
        ends = starts + config.window_seconds * 1000.0
    else:
        ends = _one_dimensional_float(
            window_end_time_ms, "window_end_time_ms", allow_empty=True
        ).copy()
        if len(ends) != len(starts):
            raise ValueError("window_end_time_ms and window_start_time_ms must align")
    if len(ends) and np.any(~np.isfinite(ends)):
        raise ValueError("window_end_time_ms must contain only finite values")
    if len(ends) and np.any(ends <= starts):
        raise ValueError("window_end_time_ms must be greater than window_start_time_ms")

    centres = (starts + ends) / 2.0
    smoothed = np.empty_like(probability)
    smoothed_logit = np.empty_like(probability)
    seizure_mask = np.zeros(len(probability), dtype=bool)
    coarse_events: list[EventInterval] = []

    stride_ms = config.stride_seconds * 1000.0
    max_gap_steps = floor(config.max_gap_seconds / config.stride_seconds + 1e-12)
    minimum_run_steps = ceil(
        config.min_event_seconds / config.stride_seconds - 1e-12
    )

    for segment in _contiguous_slices(
        centres, stride_ms * config.continuity_gap_factor
    ):
        segment_probability = probability[segment]
        segment_smoothed = rolling_median(
            segment_probability, config.median_width_windows
        )
        segment_clipped = np.clip(
            segment_smoothed,
            config.probability_clip,
            1.0 - config.probability_clip,
        )
        segment_logit = np.log(segment_clipped / (1.0 - segment_clipped))
        segment_mask = hysteresis_k_of_n(
            segment_smoothed,
            config.high_threshold,
            config.low_threshold,
            config.persistence_k,
            config.persistence_n,
        )
        segment_mask = fill_short_negative_gaps(segment_mask, max_gap_steps)
        segment_mask = remove_short_positive_runs(segment_mask, minimum_run_steps)

        smoothed[segment] = segment_smoothed
        smoothed_logit[segment] = segment_logit
        seizure_mask[segment] = segment_mask
        segment_starts = starts[segment]
        segment_ends = ends[segment]
        for run_start, run_end in _positive_runs(segment_mask):
            coarse_events.append(
                EventInterval(
                    float(segment_starts[run_start]),
                    float(segment_ends[run_end - 1]),
                )
            )

    events = _merge_event_intervals(
        coarse_events, max_gap_ms=config.max_gap_seconds * 1000.0
    )
    for array in (
        starts,
        ends,
        centres,
        probability,
        smoothed,
        smoothed_logit,
        seizure_mask,
    ):
        array.setflags(write=False)
    return WindowIntervalDecodedAnnotation(
        config=config,
        window_start_time_ms=starts,
        window_end_time_ms=ends,
        window_center_time_ms=centres,
        probability=probability,
        smoothed_probability=smoothed,
        smoothed_logit=smoothed_logit,
        seizure_mask=seizure_mask,
        coarse_events=tuple(coarse_events),
        events=tuple(events),
    )


def _merge_event_intervals(
    intervals: Iterable[EventInterval], *, max_gap_ms: float
) -> tuple[EventInterval, ...]:
    """Merge overlapping/nearby half-open event intervals deterministically."""

    if max_gap_ms < 0.0 or not isfinite(max_gap_ms):
        raise ValueError("max_gap_ms must be finite and non-negative")
    ordered = sorted(intervals, key=lambda item: (item.start_ms, item.end_ms))
    if not ordered:
        return ()
    merged: list[EventInterval] = []
    current_start = ordered[0].start_ms
    current_end = ordered[0].end_ms
    for item in ordered[1:]:
        if item.start_ms <= current_end + max_gap_ms:
            current_end = max(current_end, item.end_ms)
        else:
            merged.append(EventInterval(current_start, current_end))
            current_start = item.start_ms
            current_end = item.end_ms
    merged.append(EventInterval(current_start, current_end))
    return tuple(merged)


def match_events(
    true_events: Iterable[EventInterval | Sequence[float]],
    predicted_events: Iterable[EventInterval | Sequence[float]],
    minimum_iou: float = 0.0,
) -> tuple[EventMatch, ...]:
    """Find a strict one-to-one assignment for annotated and predicted events.

    Zero-overlap pairs are never matches.  ``minimum_iou=0`` therefore means
    any positive temporal overlap, which is the usual event-sensitivity rule.

    Qualifying pairs are assigned lexicographically: first maximize their
    count, then maximize their total IoU.  This order matters when a strict
    IoU threshold is used.  Optimising all IoUs and discarding sub-threshold
    pairs afterwards can consume a row or column that would otherwise permit
    an additional valid strict match.
    """

    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must lie in [0, 1]")
    truth = _normalise_intervals(true_events)
    prediction = _normalise_intervals(predicted_events)
    if not truth or not prediction:
        return ()

    iou = np.zeros((len(truth), len(prediction)), dtype=np.float64)
    overlap = np.zeros_like(iou)
    for true_index, annotated in enumerate(truth):
        for predicted_index, detected in enumerate(prediction):
            shared = _overlap_ms(annotated, detected)
            overlap[true_index, predicted_index] = shared
            if shared > 0.0:
                union = annotated.duration_ms + detected.duration_ms - shared
                iou[true_index, predicted_index] = shared / union

    qualifying = (overlap > 0.0) & (iou >= minimum_iou)
    if not qualifying.any():
        return ()

    # Add one dummy row per prediction and one dummy column per annotation.
    # A full square assignment can then leave either side unmatched at zero
    # cost.  Each qualifying real edge receives a reward whose cardinality
    # term dominates every possible total-IoU tie-break, so the Hungarian
    # solution is maximum-cardinality first and maximum-IoU second.
    n_truth, n_prediction = iou.shape
    maximum_possible_matches = min(n_truth, n_prediction)
    cardinality_reward = float(maximum_possible_matches + 1)
    assignment_cost = np.zeros(
        (n_truth + n_prediction, n_prediction + n_truth), dtype=np.float64
    )
    assignment_cost[:n_truth, :n_prediction][qualifying] = -(
        cardinality_reward + iou[qualifying]
    )
    true_indices, predicted_indices = linear_sum_assignment(assignment_cost)
    matches: list[EventMatch] = []
    for true_index, predicted_index in zip(true_indices, predicted_indices):
        if true_index >= n_truth or predicted_index >= n_prediction:
            continue
        if not qualifying[true_index, predicted_index]:
            continue
        pair_iou = float(iou[true_index, predicted_index])
        annotated = truth[int(true_index)]
        detected = prediction[int(predicted_index)]
        shared = float(overlap[true_index, predicted_index])
        matches.append(
            EventMatch(
                true_index=int(true_index),
                predicted_index=int(predicted_index),
                overlap_ms=shared,
                iou=pair_iou,
                dice=2.0
                * shared
                / (annotated.duration_ms + detected.duration_ms),
                annotated_coverage=shared / annotated.duration_ms,
                onset_error_seconds=(detected.start_ms - annotated.start_ms) / 1000.0,
                offset_error_seconds=(detected.end_ms - annotated.end_ms) / 1000.0,
                duration_error_seconds=(
                    detected.duration_ms - annotated.duration_ms
                )
                / 1000.0,
                duration_ratio=detected.duration_ms / annotated.duration_ms,
            )
        )
    return tuple(sorted(matches, key=lambda item: item.true_index))


def evaluate_event_annotations(
    true_events: Iterable[EventInterval | Sequence[float]],
    predicted_events: Iterable[EventInterval | Sequence[float]],
    recording_duration_seconds: float,
    minimum_iou: float = 0.0,
) -> EventEvaluation:
    """Compute clinical event and boundary metrics for one recording."""

    if recording_duration_seconds <= 0.0 or not isfinite(recording_duration_seconds):
        raise ValueError("recording_duration_seconds must be finite and positive")
    truth = _normalise_intervals(true_events)
    prediction = _normalise_intervals(predicted_events)
    matches = match_events(truth, prediction, minimum_iou=minimum_iou)
    assessments = _assess_true_events(truth, prediction)
    strict_match_counts = {
        threshold: len(match_events(truth, prediction, minimum_iou=threshold))
        for threshold in (0.1, 0.3, 0.5)
    }
    merged_predicted_events = _count_merged_predicted_events(truth, prediction)
    metrics = _summarise_evaluations(
        n_true=len(truth),
        n_predicted=len(prediction),
        matches=matches,
        assessments=assessments,
        recording_duration_seconds=recording_duration_seconds,
        strict_match_counts=strict_match_counts,
        merged_predicted_events=merged_predicted_events,
    )
    return EventEvaluation(
        metrics=metrics,
        matches=matches,
        true_events=truth,
        predicted_events=prediction,
        true_event_assessments=assessments,
    )


def decoder_config_grid(
    *,
    high_thresholds: Sequence[float],
    low_thresholds: Sequence[float],
    median_width_windows: Sequence[int] = (3,),
    persistence_values: Sequence[tuple[int, int]] = ((2, 3),),
    max_gap_seconds: Sequence[float] = (2.0,),
    min_event_seconds: Sequence[float] = (3.0,),
    boundary_search_seconds: Sequence[float] = (4.0,),
    base: DecoderConfig | None = None,
) -> tuple[DecoderConfig, ...]:
    """Build a deterministic candidate grid for validation-only fitting."""

    template = base or DecoderConfig()
    candidates: list[DecoderConfig] = []
    for high, low, width, persistence, gap, minimum, search in product(
        high_thresholds,
        low_thresholds,
        median_width_windows,
        persistence_values,
        max_gap_seconds,
        min_event_seconds,
        boundary_search_seconds,
    ):
        if low >= high:
            continue
        k, n = persistence
        candidates.append(
            replace(
                template,
                high_threshold=float(high),
                low_threshold=float(low),
                median_width_windows=int(width),
                persistence_k=int(k),
                persistence_n=int(n),
                max_gap_seconds=float(gap),
                min_event_seconds=float(minimum),
                boundary_search_seconds=float(search),
            )
        )
    if not candidates:
        raise ValueError("decoder candidate grid is empty")
    return tuple(candidates)


def fit_global_decoder_config(
    validation_sequences: Iterable[ValidationSequence],
    candidates: Iterable[DecoderConfig],
    *,
    objective: str = "event_f1",
    minimum_iou: float = 0.0,
    false_positive_penalty: float = 0.0,
) -> DecoderFitResult:
    """Select one shared configuration using validation sequences only.

    ``false_positive_penalty`` subtracts the given amount from the objective
    for each false-positive event/hour.  Ties are resolved by higher mean IoU,
    then lower false-positive event rate, then the original candidate order.
    """

    sequences = tuple(validation_sequences)
    configurations = tuple(candidates)
    if not sequences:
        raise ValueError("at least one validation sequence is required")
    if not configurations:
        raise ValueError("at least one decoder candidate is required")
    if objective not in {
        "event_f1",
        "event_sensitivity",
        "mean_iou",
        "event_f1_times_mean_iou",
    }:
        raise ValueError(f"unsupported decoder-fitting objective: {objective}")
    if false_positive_penalty < 0.0 or not isfinite(false_positive_penalty):
        raise ValueError("false_positive_penalty must be finite and non-negative")

    results: list[DecoderCandidateResult] = []
    ranking: list[tuple[float, float, float, int]] = []
    for candidate_index, config in enumerate(configurations):
        evaluations = []
        for sequence in sequences:
            decoded = decode_window_probabilities(
                sequence.probabilities,
                sequence.window_start_time_ms,
                config,
            )
            evaluations.append(
                evaluate_event_annotations(
                    sequence.true_events,
                    decoded.events,
                    sequence.recording_duration_seconds,
                    minimum_iou=minimum_iou,
                )
            )
        metrics = _pool_event_evaluations(evaluations)
        objective_value = _objective_value(metrics, objective)
        score = objective_value - false_positive_penalty * float(
            metrics["false_positive_events_per_hour"]
        )
        result = DecoderCandidateResult(config=config, score=score, metrics=metrics)
        results.append(result)
        mean_iou = _finite_or_negative_infinity(metrics["mean_iou"])
        fp_rate = float(metrics["false_positive_events_per_hour"])
        ranking.append((score, mean_iou, -fp_rate, -candidate_index))

    best_index = max(range(len(results)), key=ranking.__getitem__)
    best = results[best_index]
    return DecoderFitResult(
        config=best.config,
        objective=objective,
        score=best.score,
        metrics=best.metrics,
        candidates=tuple(results),
    )


def _events_from_mask(
    centres_ms: np.ndarray, mask: np.ndarray, stride_ms: float
) -> tuple[EventInterval, ...]:
    events = []
    for start, end in _positive_runs(mask):
        events.append(
            EventInterval(
                start_ms=float(centres_ms[start]),
                end_ms=float(centres_ms[end - 1] + stride_ms),
            )
        )
    return tuple(events)


def _refine_boundaries(
    centres_ms: np.ndarray,
    smoothed_logit: np.ndarray,
    mask: np.ndarray,
    stride_ms: float,
    search_steps: int,
) -> tuple[EventInterval, ...]:
    coarse = _events_from_mask(centres_ms, mask, stride_ms)
    if not coarse or search_steps == 0 or len(centres_ms) < 2:
        return coarse

    changes = np.diff(smoothed_logit)
    runs = _positive_runs(mask)
    refined: list[EventInterval] = []
    for run_index, (start, end) in enumerate(runs):
        last = end - 1
        previous_last = runs[run_index - 1][1] - 1 if run_index else -1
        next_start = runs[run_index + 1][0] if run_index + 1 < len(runs) else len(mask)

        onset_ms = float(centres_ms[start])
        onset_left = max(1, start - search_steps, previous_last + 1)
        onset_right = min(len(centres_ms) - 1, start + search_steps, last)
        if onset_left <= onset_right:
            candidates = np.arange(onset_left, onset_right + 1)
            change_index = int(candidates[np.argmax(changes[candidates - 1])])
            if changes[change_index - 1] > 0.0:
                onset_ms = float(
                    (centres_ms[change_index - 1] + centres_ms[change_index]) / 2.0
                )

        offset_ms = float(centres_ms[last] + stride_ms)
        offset_target = last + 1
        offset_left = max(1, offset_target - search_steps, start + 1)
        offset_right = min(
            len(centres_ms) - 1,
            offset_target + search_steps,
            next_start,
        )
        if offset_left <= offset_right:
            candidates = np.arange(offset_left, offset_right + 1)
            change_index = int(candidates[np.argmin(changes[candidates - 1])])
            if changes[change_index - 1] < 0.0:
                offset_ms = float(
                    (centres_ms[change_index - 1] + centres_ms[change_index]) / 2.0
                )

        # Independent searches can select different fluctuations inside the
        # same negative gap.  A shared midpoint keeps neighbouring refined
        # events ordered without moving either boundary across a positive run.
        if run_index:
            left_separator = float(
                (centres_ms[previous_last] + centres_ms[start]) / 2.0
            )
            onset_ms = max(onset_ms, left_separator)
        if run_index + 1 < len(runs):
            right_separator = float(
                (centres_ms[last] + centres_ms[next_start]) / 2.0
            )
            offset_ms = min(offset_ms, right_separator)

        if offset_ms <= onset_ms:
            refined.append(coarse[run_index])
        else:
            refined.append(EventInterval(onset_ms, offset_ms))
    return tuple(refined)


def _positive_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _contiguous_slices(
    centres_ms: np.ndarray, maximum_gap_ms: float
) -> tuple[slice, ...]:
    if not len(centres_ms):
        return ()
    breaks = np.flatnonzero(np.diff(centres_ms) > maximum_gap_ms) + 1
    boundaries = np.concatenate(([0], breaks, [len(centres_ms)]))
    return tuple(
        slice(int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    )


def _assess_true_events(
    truth: tuple[EventInterval, ...], prediction: tuple[EventInterval, ...]
) -> tuple[TrueEventAssessment, ...]:
    assessments = []
    for true_index, annotated in enumerate(truth):
        intersections = []
        fragments = 0
        for detected in prediction:
            start = max(annotated.start_ms, detected.start_ms)
            end = min(annotated.end_ms, detected.end_ms)
            if end > start:
                fragments += 1
                intersections.append((start, end))
        covered = _union_duration_ms(intersections)
        assessments.append(
            TrueEventAssessment(
                true_index=true_index,
                annotated_duration_ms=annotated.duration_ms,
                covered_duration_ms=covered,
                coverage=covered / annotated.duration_ms,
                overlapping_predicted_segments=fragments,
            )
        )
    return tuple(assessments)


def _count_merged_predicted_events(
    truth: Sequence[EventInterval], prediction: Sequence[EventInterval]
) -> int:
    """Count predicted intervals spanning more than one manual event.

    This is deliberately independent of the one-to-one event assignment.  A
    single very long predicted interval can be matched to only one manual
    event while still swallowing several neighbouring events; that failure
    mode must remain visible in the clinical event audit.
    """

    return int(
        sum(
            sum(_overlap_ms(annotated, detected) > 0.0 for annotated in truth) > 1
            for detected in prediction
        )
    )


def _summarise_evaluations(
    *,
    n_true: int,
    n_predicted: int,
    matches: Sequence[EventMatch],
    assessments: Sequence[TrueEventAssessment],
    recording_duration_seconds: float,
    strict_match_counts: Mapping[float, int] | None = None,
    merged_predicted_events: int = 0,
) -> dict[str, float | int]:
    n_matched = len(matches)
    false_positive_events = n_predicted - n_matched
    event_sensitivity = _ratio_or_nan(n_matched, n_true)
    event_precision = _ratio_or_nan(n_matched, n_predicted)
    event_f1 = _ratio_or_nan(2 * n_matched, n_true + n_predicted)
    recording_hours = recording_duration_seconds / 3600.0

    ious = np.asarray([match.iou for match in matches], dtype=float)
    dices = np.asarray([match.dice for match in matches], dtype=float)
    onset = np.asarray([match.onset_error_seconds for match in matches], dtype=float)
    offset = np.asarray([match.offset_error_seconds for match in matches], dtype=float)
    duration = np.asarray(
        [match.duration_error_seconds for match in matches], dtype=float
    )
    duration_ratio = np.asarray(
        [match.duration_ratio for match in matches], dtype=float
    )
    coverage = np.asarray([item.coverage for item in assessments], dtype=float)
    annotated_duration = float(
        sum(item.annotated_duration_ms for item in assessments)
    )
    covered_duration = float(sum(item.covered_duration_ms for item in assessments))
    fragment_counts = np.asarray(
        [item.overlapping_predicted_segments for item in assessments], dtype=float
    )
    coverage_zero = coverage <= 1e-12
    coverage_low = coverage < 0.5
    event_durations = np.asarray(
        [item.annotated_duration_ms for item in assessments], dtype=float
    )
    long_events = event_durations >= 30_000.0
    long_zero = long_events & coverage_zero
    strict_match_counts = strict_match_counts or {}

    output: dict[str, float | int] = {
        "true_events": int(n_true),
        "predicted_events": int(n_predicted),
        "matched_events": int(n_matched),
        "false_positive_events": int(false_positive_events),
        "recording_duration_hours": float(recording_hours),
        "event_sensitivity": event_sensitivity,
        "event_precision": event_precision,
        "event_f1": event_f1,
        "false_positive_events_per_hour": float(
            false_positive_events / recording_hours
        ),
        "mean_iou": _mean_or_nan(ious),
        "median_iou": _median_or_nan(ious),
        "mean_dice": _mean_or_nan(dices),
        "median_dice": _median_or_nan(dices),
        "annotated_event_coverage": _ratio_or_nan(
            covered_duration, annotated_duration
        ),
        "mean_annotated_event_coverage": _mean_or_nan(coverage),
        "median_annotated_event_coverage": _median_or_nan(coverage),
        "mean_onset_error_seconds": _mean_or_nan(onset),
        "median_onset_error_seconds": _median_or_nan(onset),
        "mean_absolute_onset_error_seconds": _mean_or_nan(np.abs(onset)),
        "median_absolute_onset_error_seconds": _median_or_nan(np.abs(onset)),
        "mean_offset_error_seconds": _mean_or_nan(offset),
        "median_offset_error_seconds": _median_or_nan(offset),
        "mean_absolute_offset_error_seconds": _mean_or_nan(np.abs(offset)),
        "median_absolute_offset_error_seconds": _median_or_nan(np.abs(offset)),
        "mean_duration_error_seconds": _mean_or_nan(duration),
        "median_duration_error_seconds": _median_or_nan(duration),
        "mean_absolute_duration_error_seconds": _mean_or_nan(np.abs(duration)),
        "median_absolute_duration_error_seconds": _median_or_nan(np.abs(duration)),
        "mean_duration_ratio": _mean_or_nan(duration_ratio),
        "median_duration_ratio": _median_or_nan(duration_ratio),
        "mean_predicted_segments_per_true_event": _mean_or_nan(fragment_counts),
        "fragmented_true_events": int(np.sum(fragment_counts > 1.0)),
        "fragmentation_rate": _ratio_or_nan(
            int(np.sum(fragment_counts > 1.0)), n_true
        ),
        "mean_excess_fragments_per_true_event": _mean_or_nan(
            np.maximum(fragment_counts - 1.0, 0.0)
        ),
        "zero_coverage_events": int(np.sum(coverage_zero)),
        "zero_coverage_rate": _ratio_or_nan(int(np.sum(coverage_zero)), n_true),
        "low_coverage_events_lt_0p5": int(np.sum(coverage_low)),
        "low_coverage_rate_lt_0p5": _ratio_or_nan(int(np.sum(coverage_low)), n_true),
        "long_events_ge_30s": int(np.sum(long_events)),
        "long_events_ge_30s_zero_coverage": int(np.sum(long_zero)),
        "long_events_ge_30s_zero_coverage_rate": _ratio_or_nan(
            int(np.sum(long_zero)), int(np.sum(long_events))
        ),
        "merged_predicted_events": int(merged_predicted_events),
        "merged_prediction_rate": _ratio_or_nan(
            int(merged_predicted_events), n_predicted
        ),
    }
    for threshold, label in ((0.1, "0p1"), (0.3, "0p3"), (0.5, "0p5")):
        matched = int(strict_match_counts.get(threshold, 0))
        output[f"events_iou_ge_{label}"] = matched
        output[f"sensitivity_iou_ge_{label}"] = _ratio_or_nan(matched, n_true)
        output[f"precision_iou_ge_{label}"] = _ratio_or_nan(matched, n_predicted)
        output[f"f1_iou_ge_{label}"] = _ratio_or_nan(2 * matched, n_true + n_predicted)
    return output


def _pool_event_evaluations(
    evaluations: Sequence[EventEvaluation],
) -> dict[str, float | int]:
    return _summarise_evaluations(
        n_true=sum(len(item.true_events) for item in evaluations),
        n_predicted=sum(len(item.predicted_events) for item in evaluations),
        matches=tuple(match for item in evaluations for match in item.matches),
        assessments=tuple(
            assessment
            for item in evaluations
            for assessment in item.true_event_assessments
        ),
        recording_duration_seconds=sum(
            float(item.metrics["recording_duration_hours"]) * 3600.0
            for item in evaluations
        ),
        strict_match_counts={
            threshold: sum(
                int(item.metrics.get(f"events_iou_ge_{label}", 0))
                for item in evaluations
            )
            for threshold, label in ((0.1, "0p1"), (0.3, "0p3"), (0.5, "0p5"))
        },
        merged_predicted_events=sum(
            int(item.metrics.get("merged_predicted_events", 0))
            for item in evaluations
        ),
    )


def _objective_value(metrics: dict[str, float | int], objective: str) -> float:
    if objective == "event_f1_times_mean_iou":
        f1 = _finite_or_negative_infinity(metrics["event_f1"])
        iou = _finite_or_negative_infinity(metrics["mean_iou"])
        return f1 * iou if f1 >= 0.0 and iou >= 0.0 else float("-inf")
    return _finite_or_negative_infinity(metrics[objective])


def _normalise_intervals(
    intervals: Iterable[EventInterval | Sequence[float]],
) -> tuple[EventInterval, ...]:
    output = []
    for interval in intervals:
        if isinstance(interval, EventInterval):
            output.append(interval)
            continue
        values = tuple(interval)
        if len(values) != 2:
            raise ValueError("each event interval must contain start_ms and end_ms")
        output.append(EventInterval(float(values[0]), float(values[1])))
    return tuple(output)


def _overlap_ms(left: EventInterval, right: EventInterval) -> float:
    return max(0.0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def _union_duration_ms(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _one_dimensional_float(
    values: Sequence[float] | np.ndarray, name: str, *, allow_empty: bool
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and not len(array):
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _one_dimensional_bool(values: Sequence[bool] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array.astype(bool, copy=False)


def _ratio_or_nan(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _finite_or_negative_infinity(value: float | int) -> float:
    numeric = float(value)
    return numeric if isfinite(numeric) else float("-inf")


__all__ = [
    "DecodedAnnotation",
    "WindowIntervalDecodedAnnotation",
    "DecoderCandidateResult",
    "DecoderConfig",
    "DecoderFitResult",
    "EventEvaluation",
    "EventInterval",
    "EventMatch",
    "TrueEventAssessment",
    "ValidationSequence",
    "decode_window_probabilities",
    "decode_window_intervals",
    "decoder_config_grid",
    "evaluate_event_annotations",
    "fill_short_negative_gaps",
    "fit_global_decoder_config",
    "hysteresis_k_of_n",
    "match_events",
    "remove_short_positive_runs",
    "rolling_median",
]
