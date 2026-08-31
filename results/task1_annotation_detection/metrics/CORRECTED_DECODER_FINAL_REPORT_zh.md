# Corrected decoder v1 audit result

This is a separate output line; historical v2 directories and task-2 directories were not modified.

## Protocol and denominator

- Combined protocol channel-events: 6,412 rows (train 4,495; val 960; test 957).
- Formal continuous scoring excludes only events not fully contained in one physical decoder segment: validation 954, test 953, full 6,368.
- The clinical atlas keeps all 957 held-out test channel-events in its visual denominator; the four formal-gate exclusions are retained as explicit zero-coverage audit rows. The summary figure footer now reports 957, not the historical 188-event denominator.
- Every formal assessment now has a unique (animal, session, channel, event_id) key; the previous duplicate-fragment issue is removed.
- E77/CH7 remains excluded by the frozen protocol.

## Corrected decoder definition

- Each positive decision preserves the complete 4-s physical model-window interval; overlapping positive windows are unioned.
- Segmentation uses physical/QC continuity and context IDs only; no event/label boundary is used.
- One global operating point per model is selected on validation only.
- Test and full-recording metrics are held out from selection.
- Short one-window repair settings are excluded; candidate grid has high thresholds 0.70/0.85/0.95, 2/3 or 3/5 persistence, no artificial gap bridge, and 3 or 6 s minimum run.

## Held-out test summary

| model | sensitivity | precision | event F1 | mean IoU | coverage | zero-coverage | long zero | FP/channel-hour | merged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eegwavenet_sc | 0.782 | 0.297 | 0.430 | 0.792 | 0.612 | 190 | 21 | 17.21 | 3 |
  - selected config: high=0.7, low=0.1, median=3, persistence=2/3, min=3.0s, gap=0.0s
| physio_sequence_no_aux | 0.894 | 0.215 | 0.347 | 0.781 | 0.799 | 61 | 6 | 30.27 | 18 |
  - selected config: high=0.7, low=0.1, median=3, persistence=2/3, min=3.0s, gap=0.0s
| physio_sequence_stats | 0.885 | 0.241 | 0.378 | 0.782 | 0.743 | 59 | 8 | 25.91 | 18 |
  - selected config: high=0.7, low=0.1, median=3, persistence=3/5, min=3.0s, gap=0.0s

## Interpretation

- The corrected interval support raises mean IoU and makes boundary plots physically interpretable, but it does not make the classifier clinically deployment-ready. Residual zero-coverage events and label-external candidates still require review.
- The physio_sequence_stats row above was regenerated from the current `../models/physio_sequence_stats/events_corrected_v1_test/event_metrics.json` (true-events=953); an earlier concurrent write produced a stale near-zero row in this report and must not be used.
- The old figures remain valid only as historical evidence for the old decoder. Any figure claiming corrected onset/offset or corrected event counts must use `../event_atlas/` in this release.
- `../models/<model>/events_corrected_v1_test/` contains predicted intervals, one-to-one matches and true-event coverage; `../models/<model>/events_corrected_v1_full_recording/` is descriptive full-trace output.
