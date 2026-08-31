# Corrected interval-aware held-out event atlas

This folder contains the frozen corrected-v1 test-set visual audit. Each valid EEG channel is an independent model stream; E77/CH7 remains excluded by the strict protocol. The displayed intervals and probabilities are rendered from the validation-selected `events_corrected_v1_test` outputs.

The release is self-contained. Read `corrected_atlas_manifest.json` and `DENOMINATOR_NOTE_zh.txt` for the atlas/formal denominators, and read `../metrics/CURRENT_INDEX_zh.md` plus `../metrics/CORRECTED_DECODER_FINAL_REPORT_zh.md` for the split, purge, decoder and alignment audit. No additional external audit file is required.

## Reading order

1. Open `event_atlas_summary.png` for the category counts.
2. Open `<model>/example_<category>.png` for a one-event example.
3. Open the corresponding category PDF for every event/candidate.
4. Use `event_atlas_index.csv` to locate an event by model, ID, PDF page and panel.

## Definitions

- Coverage is the fraction of a manual interval covered by the union of all overlapping decoded model intervals.
- Primary classes are mutually exclusive: coverage >=0.9; 0.5-0.9; 0-0.5; and coverage=0.
- `>=30 s missed` is a subset of complete misses.
- A count of zero for `>=30 s missed` means no held-out manual event lasting at least 30 s was completely missed under this decoder; it is not an omitted category.
- Boundary shift is a subset of detected events whose prediction-envelope onset or offset differs from the manual boundary by more than 2 s, the half-window temporal uncertainty.
- Label-external predictions have zero positive-time overlap with every manual label in the same animal, recording and channel. They are review candidates, not proof of either model error or missing labels.
- Maximum pair IoU is descriptive only. Coverage classification does not use the previous Hungarian IoU>0 matching rule.
- Raw and smoothed probabilities use the split-safe test trace. The orange decoded mask is reconstructed with the frozen validation-selected decoder.
- EEG amplitude is relative to the training-background median/MAD and remains a.u.; ADC-to-microvolt calibration is unavailable.

## Counts

| model | full | partial | low | missed | >=30 s missed | boundary shift | label-external |
|---|---:|---:|---:|---:|---:|---:|---:|
| EEGWaveNet-SC | 607 | 127 | 31 | 192 | 21 | 145 | 1685 |
| Physio-Sequence | 813 | 68 | 14 | 62 | 6 | 117 | 2933 |
| Physio-Sequence + statistics | 805 | 70 | 22 | 60 | 8 | 136 | 2484 |

## Frozen decoder settings

```json
{
  "eegwavenet_sc": {
    "high_threshold": 0.7,
    "low_threshold": 0.1,
    "median_width_windows": 3,
    "persistence_k": 2,
    "persistence_n": 3,
    "max_gap_seconds": 0.0,
    "min_event_seconds": 3.0,
    "boundary_search_seconds": 0.0,
    "window_seconds": 4.0,
    "stride_seconds": 1.0,
    "probability_clip": 1e-06,
    "continuity_gap_factor": 1.5
  },
  "physio_sequence_no_aux": {
    "high_threshold": 0.7,
    "low_threshold": 0.1,
    "median_width_windows": 3,
    "persistence_k": 2,
    "persistence_n": 3,
    "max_gap_seconds": 0.0,
    "min_event_seconds": 3.0,
    "boundary_search_seconds": 0.0,
    "window_seconds": 4.0,
    "stride_seconds": 1.0,
    "probability_clip": 1e-06,
    "continuity_gap_factor": 1.5
  },
  "physio_sequence_stats": {
    "high_threshold": 0.7,
    "low_threshold": 0.1,
    "median_width_windows": 3,
    "persistence_k": 3,
    "persistence_n": 5,
    "max_gap_seconds": 0.0,
    "min_event_seconds": 3.0,
    "boundary_search_seconds": 0.0,
    "window_seconds": 4.0,
    "stride_seconds": 1.0,
    "probability_clip": 1e-06,
    "continuity_gap_factor": 1.5
  }
}
```
