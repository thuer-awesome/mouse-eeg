# Task 2 formal PG36 corrected figures

This is the canonical visual package for the pooled morphology taxonomy. All formal figures use PG01–PG36. One observation is one independent event×channel point. CH7 and CH8 can be displayed in the same mouse panel, but they are never concatenated or treated as a paired classifier input.

## Figure semantics

- `01_per_mouse_pg_pca`: each mouse's observed PG subset, with both valid channels as independent markers and every present PG label annotated.
- `02_per_mouse_pg_waveforms`: one row per observed PG category in that mouse; CH7 and CH8 columns are independent evidence.
- `03_global_pg_centroid_pca`: exactly 36 points, one robust feature centroid per PG category, plus nearest-centroid separation.
- `04_pg_support_presence`: full PG01–PG36 × mouse/channel support matrix; zero means not observed in that stream.
- `06_classifier_pr_roc_all36`: all five models, all 36 classes, one-vs-rest PR and ROC small multiples.

The 36-class union is not expected to appear in every single mouse. Most PG categories are animal-specific; the support matrix makes that explicit. Pxx/Lxx are exploratory local labels and are not used in these formal figures.
