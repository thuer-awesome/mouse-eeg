# Corrected decoder audit v2

This is a new interval-aware audit line. Historical model/event directories and figures were not overwritten.

- Each positive decision retains the physical EEG window `[window_start, window_end)` (4 s), then overlapping/adjacent positive windows are unioned.
- Decoder operating point is fitted on validation only; held-out test is never used for selection.
- Test clinical atlas denominator: all test channel-events (957 in the combined protocol).
- Formal event metrics separately report the fully observable denominator and excluded events.
- `../models/<model>/events_corrected_v1_<scope>/decoded_tracks.parquet` is the source for the corrected waveform/probability/state atlas.
- `external` candidates mean zero positive-time overlap with the current manual labels on that channel; they require clinical review.

The training run's `events_baseline` name is retained in `RUN_SUMMARY.json` as lineage metadata. The release-level corrected interval-aware figures and formal event table are the authoritative outputs for this bundle.
