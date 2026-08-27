from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from mouse_sleep.epilepsy.data import read_events, merge_closed_events


def label_qc(session_manifest: str | Path | pd.DataFrame, output: str | Path) -> Path:
    frame = pd.read_csv(session_manifest) if not isinstance(session_manifest, pd.DataFrame) else session_manifest.copy()
    rows = []
    for row in frame.to_dict("records"):
        events = read_events(row.get("label_files", "")); merged = merge_closed_events(events)
        status = events.get("status", pd.Series(dtype=str)).astype(str).str.lower()
        durations = ((events["end_time_ms"] - events["start_time_ms"]) / 1000.0).loc[status.eq("closed")]
        rows.append({"animal_id": row["animal_id"], "session_id": row["session_id"], "label_files": len([x for x in str(row.get("label_files", "")).split(";") if x and x != "nan"]), "rows": len(events), "closed": int(status.eq("closed").sum()), "open": int(status.eq("open").sum()), "orphan_end": int(status.eq("orphan_end").sum()), "point": int(status.eq("point").sum()), "merged_events": len(merged), "median_closed_duration_s": float(durations.median()) if len(durations) else None, "max_closed_duration_s": float(durations.max()) if len(durations) else None, "closed_over_300s": int((durations > 300).sum()), "requires_physiology_review": bool((durations > 300).any() or ~status.isin(["closed"]).all()), "reviewed_scope": row.get("reviewed_scope", "all"), "reviewed_shards": row.get("reviewed_shards", "")})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"); return output


def cache_qc(cache_manifest: str | Path, output: str | Path) -> Path:
    frame = pd.read_csv(cache_manifest); rows = []
    for row in frame.to_dict("records"):
        root = Path(row["cache_dir"]); x = np.load(root / "signals.npy", mmap_mode="r"); y = np.load(root / "labels.npy", mmap_mode="r")
        pos = np.asarray(x)[y == 1] if (y == 1).any() else np.empty((0, 2, x.shape[-1]))
        rows.append({**row, "signal_min": float(np.nanmin(x)), "signal_max": float(np.nanmax(x)), "nan_count": int(np.isnan(x).sum()), "positive_rms_ch7": float(np.sqrt(np.mean(pos[:, 0] ** 2))) if len(pos) else None, "positive_rms_ch8": float(np.sqrt(np.mean(pos[:, 1] ** 2))) if len(pos) else None})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig"); return output
