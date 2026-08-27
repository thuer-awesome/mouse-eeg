#!/usr/bin/env python
"""Train frozen morphology-category classifiers on independent EEG channels.

The taxonomy is read-only input from ``04_label_freeze``.  Splits remain at
clinical-event level, and every model receives one channel without animal or
channel identity.  EEGNetv4 and EEGConformer are imported from the pinned
Braindecode implementation; EEGWaveNet-SC and PhysioSequenceNet are the
project's existing architectures.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from mouse_sleep.epilepsy.single_channel_models import build_single_channel_model

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

try:
    from braindecode.models import EEGConformer, EEGNetv4
except Exception as exc:  # pragma: no cover - audited at runtime
    EEGConformer = EEGNetv4 = None
    _BRAINECODE_IMPORT_ERROR = repr(exc)
else:
    _BRAINECODE_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output" / "epilepsy_program_20260821" / "02_morphology_taxonomy" / "all_mice_old_plus_new" / "strict_all_channel_v1"


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publication bundle: editable SVG/PDF and a 300-dpi PNG preview.
    for ext, opts in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(path.with_suffix("." + ext), bbox_inches="tight", facecolor="white", **opts)
    plt.close(fig)


class ChannelWindowDataset(Dataset):
    """Lazy raw-cache loader with event-contained 21-window context."""

    def __init__(self, rows: pd.DataFrame, context_rows: pd.DataFrame, sequence_length: int = 21):
        self.rows = rows.reset_index(drop=True)
        self.sequence_length = int(sequence_length); self.half = self.sequence_length // 2
        self.lookup: dict[tuple[str, str, str, str], pd.DataFrame] = {}
        for key, g in context_rows.groupby(["animal_id", "session_id", "event_id", "channel_name"], sort=False):
            self.lookup[key] = g.sort_values(["cache_dir", "cache_local_index", "start_time_ms"]).reset_index(drop=True)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int: return len(self.rows)

    def _signals(self, path: str) -> np.ndarray:
        if path in self._cache:
            arr = self._cache.pop(path); self._cache[path] = arr; return arr
        arr = np.load(Path(path) / "signals.npy", mmap_mode="r")
        self._cache[path] = arr
        while len(self._cache) > 24: self._cache.popitem(last=False)
        return arr

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med)) * 1.4826
        return np.clip(np.nan_to_num((x - med) / max(float(mad), 1e-3), nan=0.0), -20.0, 20.0).astype(np.float32)

    def _load_row(self, row: pd.Series) -> np.ndarray:
        arr = self._signals(str(row.cache_dir)); ci = 0 if str(row.channel_name).upper() == "CH7" else 1
        return self._norm(np.asarray(arr[int(row.cache_local_index), ci], dtype=np.float32))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        key = (str(row.animal_id), str(row.session_id), str(row.event_id), str(row.channel_name))
        group = self.lookup.get(key)
        center = self._load_row(row)
        seq = np.zeros((self.sequence_length, center.shape[0]), dtype=np.float32); mask = np.ones(self.sequence_length, dtype=bool)
        seq[self.half] = center; mask[self.half] = False
        if group is not None and len(group):
            # nearest complete windows in this event and cache; absent context
            # is padded rather than borrowed from another event/split.
            values = group.cache_local_index.to_numpy(int); target = int(row.cache_local_index)
            pos = int(np.argmin(np.abs(values - target)))
            for j in range(self.sequence_length):
                q = pos + j - self.half
                if q < 0 or q >= len(group): continue
                candidate = group.iloc[q]
                if str(candidate.cache_dir) != str(row.cache_dir): continue
                if abs(int(candidate.cache_local_index) - target) > self.half: continue
                seq[j] = self._load_row(candidate); mask[j] = False
            seq[self.half] = center; mask[self.half] = False
        return {"signals": torch.from_numpy(seq[:, None, :]), "mask": torch.from_numpy(mask), "label": int(row.global_category), "index": index}


class BraindecodeAdapter(nn.Module):
    uses_sequence_context = False
    def __init__(self, name: str, classes: int, samples: int):
        super().__init__(); self.name = name; self.output_is_log_probability = name == "eegconformer"
        if name == "eegnetv4":
            if EEGNetv4 is None: raise RuntimeError(f"Braindecode unavailable: {_BRAINECODE_IMPORT_ERROR}")
            self.model = EEGNetv4(n_chans=1, n_outputs=classes, n_times=samples, F1=8, D=2, F2=16, kernel_length=64, drop_prob=.25)
        elif name == "eegconformer":
            if EEGConformer is None: raise RuntimeError(f"Braindecode unavailable: {_BRAINECODE_IMPORT_ERROR}")
            self.model = EEGConformer(n_chans=1, n_outputs=classes, n_times=samples, n_filters_time=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15, att_depth=6, att_heads=10, drop_prob=.5, att_drop_prob=.5, final_fc_length="auto", add_log_softmax=True)
        else: raise ValueError(name)
    def forward(self, signals: torch.Tensor, padding_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = signals[:, 0] if signals.ndim == 4 else signals
        return {"logits": self.model(x)}


def build_model(name: str, classes: int, samples: int, sequence_length: int) -> tuple[nn.Module, bool, bool]:
    if name in {"eegnetv4", "eegconformer"}:
        m = BraindecodeAdapter(name, classes, samples); return m, False, bool(m.output_is_log_probability)
    if name == "eegwavenet_sc": return build_single_channel_model(name, classes=classes, samples_per_window=samples, sequence_length=sequence_length), False, True
    if name == "physio_tf":
        return build_single_channel_model("physio_sequence_time_frequency", classes=classes, samples_per_window=samples, sequence_length=sequence_length, include_statistics_input=False, auxiliary_statistics=False), True, False
    if name == "physio_stats":
        return build_single_channel_model("physio_sequence_stats_aux", classes=classes, samples_per_window=samples, sequence_length=sequence_length, include_statistics_input=True, auxiliary_statistics=True), True, False
    raise ValueError(name)


def output_log_probs(model: nn.Module, out: dict[str, torch.Tensor], log_model: bool) -> torch.Tensor:
    logits = out["logits"]
    return logits if log_model else F.log_softmax(logits, dim=1)


def select_rows(df: pd.DataFrame, split: str, max_per_event: int, seed: int) -> pd.DataFrame:
    d = df.loc[(df.split == split) & df.global_category.notna()].copy()
    # event-balanced cap; every channel remains an independent observation.
    split_seed = {"train": 11, "val": 23, "test": 37}.get(str(split), 0)
    rng = np.random.default_rng(seed + split_seed); chunks=[]
    for _, g in d.groupby(["animal_id","session_id","event_id","channel_name"], sort=False):
        if len(g) <= max_per_event: chunks.append(g)
        else:
            ix = np.linspace(0, len(g)-1, max_per_event).round().astype(int); chunks.append(g.iloc[np.unique(ix)])
    return pd.concat(chunks, ignore_index=True) if chunks else d.iloc[:0]


def split_label_frame(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/validation/test rows after checking event-level isolation."""
    required = {"split", "animal_id", "session_id", "event_id", "global_category"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"PG36 label table is missing columns: {sorted(missing)}")
    frame = labels.loc[labels["global_category"].notna()].copy()
    frame["global_category"] = frame["global_category"].astype(int)
    frame["split"] = frame["split"].astype(str).str.lower()
    unknown = sorted(set(frame["split"]) - {"train", "val", "test"})
    if unknown:
        raise ValueError(f"unknown split values: {unknown}")
    event_keys = ["animal_id", "session_id", "event_id"]
    violations = frame.groupby(event_keys, dropna=False)["split"].nunique()
    if bool(violations.gt(1).any()):
        raise ValueError("an event appears in more than one train/val/test split")
    parts = tuple(frame.loc[frame["split"].eq(name)].copy() for name in ("train", "val", "test"))
    if any(part.empty for part in parts):
        raise ValueError("PG36 labels must contain non-empty train, val and test splits")
    return parts  # type: ignore[return-value]


def metrics(y: np.ndarray, p: np.ndarray, classes: int, label_prefix: str = "G") -> tuple[dict[str,float], pd.DataFrame, pd.DataFrame]:
    pred = p.argmax(1); cm = confusion_matrix(y, pred, labels=np.arange(classes));
    result = {"n":int(len(y)),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y,pred,average="weighted",zero_division=0))}
    ybin=np.eye(classes,dtype=int)[y]; aps=[]; rocs=[]
    for c in range(classes):
        if ybin[:,c].sum() and ybin[:,c].sum()<len(y):
            aps.append(average_precision_score(ybin[:,c],p[:,c])); rocs.append(roc_auc_score(ybin[:,c],p[:,c]))
    result["macro_auprc"]=float(np.mean(aps)) if aps else float("nan"); result["macro_auroc"]=float(np.mean(rocs)) if rocs else float("nan")
    labels = [f"{label_prefix}{i+1:02d}" for i in range(classes)]
    per=[]
    for c in range(classes):
        tp=cm[c,c]; support=cm[c].sum(); per.append({"class":c,"label":labels[c],"support":int(support),"precision":float(tp/max(1,cm[:,c].sum())),"recall":float(tp/max(1,support)),"f1":float(2*tp/max(1,2*tp+(cm[:,c].sum()-tp)+(support-tp)))})
    return result,pd.DataFrame(per),pd.DataFrame(cm,index=labels,columns=labels)


def run_one(name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, context_df: pd.DataFrame, out: Path, classes: int, samples: int, seq_len: int, epochs: int, batch_size: int, seed: int, max_per_event: int, device: torch.device, output_subdir: str = "05_classification", label_prefix: str = "G", learning_rate: float = 3e-4, weight_decay: float = 1e-4, early_stopping_patience: int = 3, learning_rate_patience: int = 1, num_workers: int = 0) -> dict[str, Any]:
    seed_all(seed); model, needs_seq, log_model = build_model(name, classes, samples, seq_len); model.to(device)
    train_rows=select_rows(train_df,"train",max_per_event,seed); val_rows=select_rows(val_df,"val",max_per_event,seed); test_rows=select_rows(test_df,"test",max_per_event,seed)
    # Include all eligible rows in context lookup, but no label from another event.
    train_ds=ChannelWindowDataset(train_rows,context_df,seq_len); val_ds=ChannelWindowDataset(val_rows,context_df,seq_len); test_ds=ChannelWindowDataset(test_rows,context_df,seq_len)
    workers=int(num_workers); train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=workers,pin_memory=torch.cuda.is_available()); val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=torch.cuda.is_available()); test_loader=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=torch.cuda.is_available())
    if learning_rate <= 0 or weight_decay < 0 or early_stopping_patience < 1 or learning_rate_patience < 1:
        raise ValueError("learning rate, weight decay and patience values are invalid")
    counts=np.bincount(train_rows.global_category.astype(int),minlength=classes); weights=np.sqrt(counts.sum()/np.maximum(counts,1)); weights=weights/weights.mean(); criterion=nn.NLLLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device)); opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=learning_rate_patience,min_lr=1e-6)
    hist=[]; best=-np.inf; best_epoch=0; best_state=None; patience=int(early_stopping_patience); stale=0
    def epoch(loader, train: bool):
        model.train(train); losses=[]; ys=[]; ps=[]
        for batch in loader:
            seq=batch["signals"].to(device,non_blocking=True); mask=batch["mask"].to(device,non_blocking=True); y=batch["label"].to(device)
            inp=seq if needs_seq else seq[:,seq_len//2]
            if train: opt.zero_grad(set_to_none=True)
            out=model(inp,padding_mask=mask if needs_seq else None); lp=output_log_probs(model,out,log_model); loss=criterion(lp,y)
            if train and "statistics_prediction" in out:
                target=out["statistics_target"]; sm=out.get("statistics_mask",torch.isfinite(target)); predstat=out["statistics_prediction"]; valid=sm & torch.isfinite(predstat)
                if bool(valid.any()): loss=loss+0.05*F.smooth_l1_loss(predstat[valid],target[valid])
            if train: loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            losses.append(float(loss.detach().cpu())); ys.append(y.detach().cpu().numpy()); ps.append(lp.detach().cpu().numpy())
        y=np.concatenate(ys) if ys else np.empty(0,int); p=np.concatenate(ps) if ps else np.empty((0,classes)); m,_,_=metrics(y,np.exp(p),classes,label_prefix) if len(y) else ({"macro_f1":float("nan")},None,None); return float(np.mean(losses)) if losses else float("nan"),m,y,np.exp(p)
    for ep in range(1,epochs+1):
        tr_loss,tr_m,_,_=epoch(train_loader,True); va_loss,va_m,_,_=epoch(val_loader,False); sched.step(va_m.get("macro_f1",-np.inf)); hist.append({"epoch":ep,"train_loss":tr_loss,"val_loss":va_loss,"train_macro_f1":tr_m.get("macro_f1",np.nan),"val_macro_f1":va_m.get("macro_f1",np.nan),"val_macro_auprc":va_m.get("macro_auprc",np.nan),"lr":opt.param_groups[0]["lr"]}); print(f"[{name}] epoch {ep}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_macroF1={va_m.get('macro_f1',float('nan')):.4f}",flush=True)
        score=va_m.get("macro_f1",-np.inf)
        if score>best+1e-4: best=score; best_epoch=ep; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    _,test_m,y,p=epoch(test_loader,False); _,per,cm=metrics(y,p,classes,label_prefix)
    model_out=out/output_subdir/name; model_out.mkdir(parents=True,exist_ok=True); pd.DataFrame(hist).to_csv(model_out/"history.csv",index=False); per.to_csv(model_out/"per_class_metrics.csv",index=False); cm.to_csv(model_out/"confusion_matrix.csv"); pd.DataFrame({"y_true":y,"y_pred":p.argmax(1),**{f"prob_{label_prefix}{i+1:02d}":p[:,i] for i in range(classes)}}).to_parquet(model_out/"test_predictions.parquet",index=False); torch.save({"model":name,"classes":classes,"label_prefix":label_prefix,"output_subdir":output_subdir,"state_dict":model.state_dict(),"best_epoch":best_epoch,"seed":seed},model_out/"best_model.pt"); (model_out/"metrics.json").write_text(json.dumps({**test_m,"best_epoch":best_epoch,"train_rows":len(train_rows),"val_rows":len(val_rows),"test_rows":len(test_rows),"class_counts_train":counts.tolist(),"label_prefix":label_prefix,"output_subdir":output_subdir,"learning_rate":learning_rate,"weight_decay":weight_decay,"early_stopping_patience":early_stopping_patience,"learning_rate_patience":learning_rate_patience,"num_workers":num_workers},ensure_ascii=False,indent=2),encoding="utf-8")
    # Compact diagnostic figure: learning curves + normalized confusion matrix.
    h=pd.DataFrame(hist); fig,ax=plt.subplots(1,2,figsize=(183/25.4,65/25.4),constrained_layout=True); ax[0].plot(h.epoch,h.train_loss,"o-",label="train",color="#0072B2"); ax[0].plot(h.epoch,h.val_loss,"o-",label="validation",color="#D55E00"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("NLL + auxiliary loss"); ax[0].legend(frameon=False); ax[0].set_title("Training diagnostic",loc="left",fontweight="bold"); norm=cm.div(cm.sum(axis=1).replace(0,1),axis=0); im=ax[1].imshow(norm.to_numpy(),vmin=0,vmax=1,cmap="Blues"); ax[1].set_xticks(range(classes),norm.columns); ax[1].set_yticks(range(classes),norm.index); ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True"); ax[1].set_title(f"Test confusion\nmacro-F1 {test_m['macro_f1']:.3f}",loc="left",fontweight="bold"); fig.colorbar(im,ax=ax[1],fraction=.05,pad=.04,label="row fraction"); save_fig(fig,model_out/"figure_model_diagnostics")
    return {"model":name,**test_m,"best_epoch":best_epoch,"train_rows":len(train_rows),"val_rows":len(val_rows),"test_rows":len(test_rows)}


def main(argv: list[str] | None = None) -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=DEFAULT_OUT); ap.add_argument("--label-path",type=Path,default=None,help="frozen channel-window label parquet"); ap.add_argument("--output-subdir",default="05_classification",help="classification result directory under --out"); ap.add_argument("--label-prefix",default=None,help="category prefix used in reports, e.g. PG"); ap.add_argument("--epochs",type=int,default=10); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--max-per-event",type=int,default=8); ap.add_argument("--seed",type=int,default=20260823); ap.add_argument("--models",nargs="+",default=["eegwavenet_sc","physio_tf","physio_stats","eegnetv4","eegconformer"]); ap.add_argument("--device",default="auto"); ap.add_argument("--learning-rate",type=float,default=3e-4); ap.add_argument("--weight-decay",type=float,default=1e-4); ap.add_argument("--early-stopping-patience",type=int,default=3); ap.add_argument("--learning-rate-patience",type=int,default=1); ap.add_argument("--num-workers",type=int,default=0); args=ap.parse_args(argv)
    out=args.out.resolve(); seed_all(args.seed); device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device));
    label_path = (args.label_path.resolve() if args.label_path is not None else out/"04_label_freeze"/"frozen_channel_window_labels.parquet")
    labels=pd.read_parquet(label_path); labels=labels.loc[labels.global_category.notna()].copy(); labels["global_category"]=labels.global_category.astype(int); classes=int(labels.global_category.max()+1); context=labels.copy();
    train_labels, val_labels, test_labels = split_label_frame(labels)
    inferred_prefix = "PG" if labels.get("global_label", pd.Series(dtype=str)).astype(str).str.startswith("PG").any() else "G"
    label_prefix = args.label_prefix or inferred_prefix
    summaries=[]
    for name in args.models:
        try: summaries.append(run_one(name,train_labels,val_labels,test_labels,context, out,classes,1000,21,args.epochs,args.batch_size,args.seed,max(1,args.max_per_event),device,args.output_subdir,label_prefix,args.learning_rate,args.weight_decay,args.early_stopping_patience,args.learning_rate_patience,args.num_workers))
        except Exception as exc:
            model_out=out/args.output_subdir/name; model_out.mkdir(parents=True,exist_ok=True); (model_out/"FAILED.txt").write_text(repr(exc),encoding="utf-8"); print(f"[{name}] FAILED: {exc!r}",flush=True)
    summary=pd.DataFrame(summaries); summary.to_csv(out/args.output_subdir/"model_comparison.csv",index=False,encoding="utf-8-sig")
    if not summary.empty:
        fig,ax=plt.subplots(figsize=(183/25.4,70/25.4),constrained_layout=True); x=np.arange(len(summary)); w=.18; metrics_to_plot=[("macro_f1","Macro-F1","#0072B2"),("macro_auprc","Macro-AUPRC","#D55E00"),("balanced_accuracy","Balanced accuracy","#009E73")];
        for j,(col,label,color) in enumerate(metrics_to_plot): ax.bar(x+(j-1)*w,summary[col],w,label=label,color=color)
        ax.set_xticks(x,summary.model,rotation=25,ha="right"); ax.set_ylim(0,1); ax.set_ylabel("Test score"); ax.set_title(f"Frozen {label_prefix} morphology-category classifier comparison",loc="left",fontweight="bold"); ax.legend(frameon=False,ncol=3); save_fig(fig,out/"06_figures"/args.output_subdir/"figure_4_classifier_comparison")
    # Keep the curated Chinese report separate from this machine-generated
    # run note so a future retraining cannot silently replace the audited
    # denominator/label documentation.
    (out/"07_reports"/"CLASSIFIER_TASK2_REPORT_AUTO.md").write_text("# 冻结形态类别分类器自动摘要\n\n分类标签：`04_label_freeze/frozen_channel_window_labels.parquet`；模型："+", ".join(args.models)+f"；类别数：{classes}；序列上下文：21 epoch。\n事件平衡敏感性结果写入 `05_classification/model_comparison.csv`；全窗口主结果由 `evaluate_strict_classifiers_all_windows.py` 写入 `05_classification/all_window_model_comparison.csv`。\n",encoding="utf-8")
    _json(out/"08_audit"/f"classifier_environment_{args.output_subdir}.json",{"torch":torch.__version__,"cuda":torch.cuda.is_available(),"device":str(device),"braindecode_import_error":_BRAINECODE_IMPORT_ERROR,"models_requested":args.models,"classes":classes,"sequence_length":21,"normalization":"per-window median/MAD","weights_pretrained":False,"label_path":str(label_path),"label_prefix":label_prefix,"output_subdir":args.output_subdir,"split_counts":{"train":int(len(train_labels)),"val":int(len(val_labels)),"test":int(len(test_labels))},"learning_rate":args.learning_rate,"weight_decay":args.weight_decay,"early_stopping_patience":args.early_stopping_patience,"learning_rate_patience":args.learning_rate_patience,"num_workers":args.num_workers})
    print(summary.to_string(index=False) if not summary.empty else "No model completed",flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
