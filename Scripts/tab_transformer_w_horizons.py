import os, argparse, glob, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Union, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# CONFIG
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)

# Forecast defaults
HORIZONS = [1, 3, 7]
TOP_AGENCIES = 10
TOP_VIOLATIONS = 20
TEST_FRACTION = 0.30
VAL_FRACTION  = 0.10
D_TOKEN_F     = 64
N_LAYERS_F    = 2
N_HEADS_F     = 4
DROPOUT_F     = 0.20
WEIGHT_DECAY_F= 1e-3
HUBER_BETA_D  = 0.2
HORIZON_WEIGHTS_DEFAULT = [1.3, 1.1, 1.0]

FORECAST_DATA_PATH = "../CleanData/cleaned_parking_violations_v2.csv"

# Output dirs
FORECAST_OUT = "tabtransformer_outputs"
GEO_OUT      = "geo_outputs"
os.makedirs(FORECAST_OUT, exist_ok=True)
os.makedirs(GEO_OUT, exist_ok=True)

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
)

# Holidays
country = "US"
import holidays as _hol
_holiday_calendar = _hol.CountryHoliday(country)
def is_holiday(d: pd.Timestamp) -> int:
    return int(pd.Timestamp(d).date() in _holiday_calendar)
print(f"Holidays enabled for {country}.")


# ==============================================================
#                         COMMON UTILS
# ==============================================================

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100.0

def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================
#                    FORECASTING PIPELINE
# ==============================================================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip() for c in df.columns})

def _lower_map(df: pd.DataFrame) -> Dict[str, str]:
    return {c.lower(): c for c in df.columns}

def add_time_features(frame: pd.DataFrame, index_like: pd.DatetimeIndex) -> pd.DataFrame:
    f = frame.copy(); f.index = index_like
    f["day_of_week"]  = index_like.dayofweek
    f["is_weekend"]   = (f["day_of_week"] >= 5).astype(int)
    f["is_holiday"]   = pd.Index(index_like).map(lambda d: is_holiday(pd.Timestamp(d))).astype(int)
    f["week_of_year"] = index_like.isocalendar().week.astype(int)
    f["month"]        = index_like.month
    f["day_of_month"] = index_like.day
    return f

def build_daily_matrix(df_raw: pd.DataFrame,
                       top_agencies=TOP_AGENCIES,
                       top_violations=TOP_VIOLATIONS,
                       use_recency: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = _normalize_columns(df_raw)
    L = _lower_map(df)

    # Date parsing
    if "issue_datetime" in L:
        dt_series = pd.to_datetime(df[L["issue_datetime"]], errors="coerce")
    elif "issue_date" in L:
        dt_series = pd.to_datetime(df[L["issue_date"]], errors="coerce")
    elif all(k in L for k in ["year", "month", "day"]):
        dt_series = pd.to_datetime(
            df[L["year"]].astype(str) + "-" + df[L["month"]].astype(str) + "-" + df[L["day"]].astype(str),
            errors="coerce")
    else:
        raise ValueError("Could not find a date field (issue_datetime, issue_date, or year/month/day).")

    df["_ISSUE_DATE"] = pd.to_datetime(pd.to_datetime(dt_series).dt.date)
    df = df.dropna(subset=["_ISSUE_DATE"])

    daily_counts = df.groupby("_ISSUE_DATE").size().rename("tickets").sort_index()

    # Base calendar features
    feat = pd.DataFrame(index=daily_counts.index)
    feat = add_time_features(feat, feat.index)
    feat["days_since_start"] = (feat.index - feat.index.min()).days

    # Agency one-hot (top K + OTHER)
    agency_col = None
    for cand in ["issuing_agency_short", "issuing_agency_name", "issuing_agency_code"]:
        if cand in L: agency_col = L[cand]; break
    if agency_col is not None:
        top_ag = df[agency_col].astype(str).fillna("UNKNOWN").value_counts().head(top_agencies).index
        df["_agency"] = df[agency_col].astype(str).fillna("UNKNOWN").where(df[agency_col].isin(top_ag), other="OTHER")
        agency_pivot = pd.crosstab(df["_ISSUE_DATE"], df["_agency"]).reindex(feat.index, fill_value=0)
        keep = [c for c in agency_pivot.columns if (c in top_ag) or (c == "OTHER")]
        feat = feat.join(agency_pivot[keep].add_prefix("agency_"))

    # Violation one-hot (top K + OTHER)
    viol_col = L.get("violation_code")
    if viol_col is not None:
        top_viol = df[viol_col].astype(str).fillna("UNKNOWN").value_counts().head(top_violations).index
        df["_viol"] = df[viol_col].astype(str).fillna("UNKNOWN").where(df[viol_col].isin(top_viol), other="OTHER")
        viol_pivot = pd.crosstab(df["_ISSUE_DATE"], df["_viol"]).reindex(feat.index, fill_value=0)
        keep = [c for c in viol_pivot.columns if (c in top_viol) or (c == "OTHER")]
        feat = feat.join(viol_pivot[keep].add_prefix("viol_"))

    # Optional recency lags
    if use_recency:
        s = daily_counts.reindex(feat.index)
        feat["lag_1"] = s.shift(1)
        feat["lag_3"] = s.shift(3)
        feat["lag_7"] = s.shift(7)
        feat["ma_7"]  = s.rolling(7, min_periods=1).mean()
        feat[["lag_1","lag_3","lag_7","ma_7"]] = feat[["lag_1","lag_3","lag_7","ma_7"]].fillna(0)

    # Spatial reference (for allocation only)
    spatial_ref = pd.DataFrame()
    if "latitude" in df.columns and "longitude" in df.columns:
        tmp = df.dropna(subset=["latitude", "longitude"]).copy()
        tmp["lat_grid"] = pd.to_numeric(tmp["latitude"], errors="coerce").round(3)
        tmp["lon_grid"] = pd.to_numeric(tmp["longitude"], errors="coerce").round(3)
        tmp = tmp.dropna(subset=["lat_grid", "lon_grid"])
        tmp["day_of_week"] = tmp["_ISSUE_DATE"].dt.dayofweek
        tmp["month"] = tmp["_ISSUE_DATE"].dt.month
        grp = (tmp.groupby(["day_of_week", "month", "lat_grid", "lon_grid"]).size()
                 .rename("count").reset_index())
        grp["weight"] = grp["count"] / grp.groupby(["day_of_week", "month"])['count'].transform('sum')
        spatial_ref = grp[["day_of_week", "month", "lat_grid", "lon_grid", "weight"]].copy()

    X_base = feat.sort_index()
    y_base = daily_counts.reindex(X_base.index).astype(int)
    return X_base, y_base, spatial_ref

#  Dataset + feature transform
CAT_COLUMNS = ["day_of_week", "is_weekend", "is_holiday", "week_of_year", "month", "day_of_month"]

def split_time(X: pd.DataFrame, y: pd.Series, test_fraction=TEST_FRACTION):
    n = len(X); test_size = max(1, int(np.floor(test_fraction * n))); split_idx = n - test_size
    return (X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:])

class DailyDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, horizons: List[int]):
        self.horizons = horizons
        Y, M = [], []
        for h in horizons:
            y_h = y.shift(-h)
            mask = y_h.notna().astype(int)
            Y.append(y_h.fillna(0).to_numpy(dtype=np.float32))
            M.append(mask.to_numpy(dtype=np.float32))
        self.labels = torch.tensor(np.stack(Y, axis=1), dtype=torch.float32)
        self.label_mask = torch.tensor(np.stack(M, axis=1), dtype=torch.float32)

        cats = []
        for c in CAT_COLUMNS:
            v = X[c].astype(int).to_numpy()
            if c == "month": v = v - 1
            if c == "day_of_month": v = v - 1
            if c == "week_of_year": v = np.clip(v - 1, 0, 52)
            cats.append(v)
        self.cat = torch.tensor(np.stack(cats, axis=1), dtype=torch.long)
        cont_cols = [c for c in X.columns if c not in CAT_COLUMNS]
        self.cont_cols = cont_cols
        self.cont = torch.tensor(X[cont_cols].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self): return len(self.cat)
    def __getitem__(self, idx): return self.cat[idx], self.cont[idx], self.labels[idx], self.label_mask[idx]

class FeatureTransformer:
    def __init__(self, count_prefixes=("agency_", "viol_", "lag_", "ma_")):
        self.count_prefixes = count_prefixes
        self.ds_mean = None; self.ds_std = None; self.count_cols = None
    def fit(self, X: pd.DataFrame):
        assert "days_since_start" in X.columns
        self.ds_mean = float(X["days_since_start"].mean())
        std = float(X["days_since_start"].std()); self.ds_std = std if std != 0.0 else 1.0
        self.count_cols = [c for c in X.columns if any(c.startswith(p) for p in self.count_prefixes)]
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["days_since_start"] = (X["days_since_start"] - self.ds_mean) / self.ds_std
        if self.count_cols:
            X[self.count_cols] = np.log1p(X[self.count_cols])
        return X

#  TabTransformer
class TabTransformerReg(nn.Module):
    def __init__(self, cardinals: List[int], cont_dim: int, d_token=D_TOKEN_F, n_heads=N_HEADS_F,
                 n_layers=N_LAYERS_F, dropout=DROPOUT_F, out_dim: int = 1, decoupled_heads: bool = True):
        super().__init__()
        self.decoupled = decoupled_heads
        self.embs = nn.ModuleList([nn.Embedding(card, d_token) for card in cardinals])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token*4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cont_proj = nn.Sequential(nn.Linear(cont_dim, d_token), nn.ReLU(), nn.Dropout(dropout))
        self.repr_dim = d_token * (len(cardinals) + 2)  # pooled + each token + cont
        self.norm = nn.LayerNorm(self.repr_dim)

        if self.decoupled and out_dim > 1:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.repr_dim, d_token*2), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(d_token*2, 1),
                ) for _ in range(out_dim)
            ])
        else:
            self.head = nn.Sequential(
                nn.Linear(self.repr_dim, d_token*2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_token*2, out_dim),
            )

    def forward(self, cat_tokens: torch.Tensor, cont_feats: torch.Tensor) -> torch.Tensor:
        toks = [emb(cat_tokens[:, i]) for i, emb in enumerate(self.embs)]   
        Z = torch.stack(toks, dim=1)                                   
        Z = self.encoder(Z)
        Z_pool = Z.mean(dim=1)                                           
        C = self.cont_proj(cont_feats)                                    
        X = torch.cat([Z_pool] + toks + [C], dim=1)                       
        X = self.norm(X)
        if hasattr(self, "heads"):
            outs = [head(X) for head in self.heads]
            return torch.cat(outs, dim=1)                                
        else:
            return self.head(X)                                          

def train_one_model_forecast(X_train, X_val, y_train, y_val, horizons: List[int], *,
                             batch_size=256, epochs=200, lr=1e-3, device=DEVICE,
                             log_target: bool = False, run_tag: str = None,
                             huber_beta: float = HUBER_BETA_D,
                             horizon_weights: Iterable[float] = HORIZON_WEIGHTS_DEFAULT,
                             seed: int = RANDOM_STATE):
    set_all_seeds(seed)
    ds_tr = DailyDataset(X_train, y_train, horizons)
    ds_va = DailyDataset(X_val,   y_val,   horizons)

    cardinals = [7, 2, 2, 53, 12, 31]
    cont_dim  = ds_tr.cont.shape[1]

    model = TabTransformerReg(cardinals=cardinals, cont_dim=cont_dim, out_dim=len(horizons)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY_F)

    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-5, verbose=True)
    patience, epochs_no_improve = 6, 0
    best_va = float('inf'); best_state = None

    w = torch.tensor(list(horizon_weights), dtype=torch.float32, device=device).view(1, -1)
    crit = nn.SmoothL1Loss(reduction='none', beta=huber_beta)

    history = {"epoch": [], "train_loss": [], "val_rmse": []}

    for epoch in range(1, epochs+1):
        model.train(); total = 0.0; nobs = 0
        for cat, cont, y, m in tr_loader:
            cat, cont, y, m = cat.to(device), cont.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            y_space = torch.log1p(torch.clamp(y, min=0)) if log_target else y
            pred = model(cat, cont)
            loss_raw = crit(pred, y_space)                    # [B,H]
            loss = (loss_raw * w * m).sum() / ((w * m).sum() + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            total += loss.item() * cat.size(0); nobs += cat.size(0)
        avg_tr = total / max(1, nobs)

        # validation 
        model.eval(); y_true = []; y_pred = []; mks = []
        with torch.no_grad():
            for cat, cont, y, m in va_loader:
                pred = model(cat.to(device), cont.to(device)).cpu().numpy()
                Y = (np.log1p(np.clip(y.numpy(), 0, None))) if log_target else y.numpy()
                y_true.append(Y); y_pred.append(pred); mks.append(m.numpy())
        Yt = np.concatenate(y_true); Yp = np.concatenate(y_pred); Mk = np.concatenate(mks)
        rmse = np.sqrt(((Yt - Yp)**2 * Mk).sum() / (Mk.sum() + 1e-8))

        print(f"Epoch {epoch:02d}  train_loss={avg_tr:.4f}  val_RMSE{'_log' if log_target else ''}={rmse:.4f}")
        history["epoch"].append(epoch); history["train_loss"].append(avg_tr); history["val_rmse"].append(rmse)

        scheduler.step(rmse)
        if rmse < best_va - 1e-4:
            best_va = rmse; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val_RMSE{'_log' if log_target else ''}={best_va:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # save training curve + CSV 
    tag = run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(history["epoch"], history["train_loss"], label="train_loss (Huber)")
    ax1.set_ylabel("Train Loss (Huber)", color="tab:blue"); ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_rmse"], color="tab:orange",
             label=f"val_RMSE{' (log)' if log_target else ''}")
    ax2.set_ylabel("Val RMSE", color="tab:orange"); ax2.tick_params(axis="y", labelcolor="tab:orange")
    fig.suptitle(f"Training vs Validation (seed={seed})"); fig.tight_layout()
    curve_path = os.path.join(FORECAST_OUT, f"training_curve_dual_{tag}_seed{seed}.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved training curve to {curve_path}")

    hist_path = os.path.join(FORECAST_OUT, f"training_history_{tag}_seed{seed}.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"Saved training history to {hist_path}")

    return model

ModelOrEnsemble = Union[nn.Module, List[nn.Module]]

def _predict_batch_reg(models: ModelOrEnsemble, cat: torch.Tensor, cont: torch.Tensor) -> np.ndarray:
    if isinstance(models, list):
        outs = [m(cat, cont).detach().cpu().numpy() for m in models]
        return np.mean(np.stack(outs, axis=0), axis=0)
    else:
        return models(cat, cont).detach().cpu().numpy()

def fit_scale_only_calibration(pred_raw: np.ndarray, y_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fit per-horizon scale 'a' minimizing ||y - a*pred||^2 on the validation set.
    Returns a of shape [H].
    """
    H = pred_raw.shape[1]; a = np.ones(H, dtype=np.float64)
    for j in range(H):
        m = mask[:, j] > 0
        if m.sum() < 5: continue
        x = pred_raw[m, j].astype(np.float64); y = y_raw[m, j].astype(np.float64)
        num = (x * y).sum(); den = (x * x).sum() + 1e-12
        a[j] = num / den
    return a

@torch.no_grad()
def evaluate_forecast(
    models: ModelOrEnsemble,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizons: List[int],
    *,
    device=DEVICE,
    log_target: bool = False,
    run_tag: str | None = None,
    expm1_cap_log: float | None = None,
    scale_calibration: Optional[np.ndarray] = None,
) -> pd.DataFrame:

    ds_te = DailyDataset(X_test, y_test, horizons)
    loader = DataLoader(ds_te, batch_size=512, shuffle=False, num_workers=0)

    Yt_list, Yp_list, Mk_list = [], [], []
    for cat, cont, y, m in loader:
        pred = _predict_batch_reg(models, cat.to(device), cont.to(device))  # [B,H]
        Yt_list.append(y.numpy()); Yp_list.append(pred); Mk_list.append(m.numpy())
    Yt = np.concatenate(Yt_list); Yp = np.concatenate(Yp_list); Mk = np.concatenate(Mk_list)

    # to raw space
    if log_target:
        cap_log = float(expm1_cap_log) if expm1_cap_log is not None else 15.0
        Yp_raw = np.expm1(np.clip(Yp, None, cap_log))
    else:
        Yp_raw = Yp

    if scale_calibration is not None:
        Yp_raw = Yp_raw * scale_calibration.reshape(1, -1)

    rows = []
    for j, h in enumerate(horizons):
        mask = Mk[:, j] > 0
        yt, yp = Yt[mask, j], Yp_raw[mask, j]
        rmse = root_mean_squared_error(yt, yp)
        mae  = mean_absolute_error(yt, yp)
        sm   = smape(yt, yp)
        r2   = r2_score(yt, yp)
        print(f"\n==== Horizon t+{h} ====")
        print(f" Test RMSE : {rmse:,.2f}")
        print(f" Test MAE  : {mae:,.2f}")
        print(f" Test sMAPE: {sm:,.2f}%")
        print(f" Test R^2  : {r2:,.4f}")
        rows.append({"horizon": h, "RMSE": rmse, "MAE": mae, "sMAPE_percent": sm, "R2": r2})

    # log metrics
    if log_target:
        Yt_log = np.log1p(np.clip(Yt, 0, None))
        for j, h in enumerate(horizons):
            mask = Mk[:, j] > 0
            rmse_log = root_mean_squared_error(Yt_log[mask, j], Yp[mask, j])
            mae_log  = mean_absolute_error(Yt_log[mask, j],  Yp[mask, j])
            print(f"(log) t+{h}: RMSE_log={rmse_log:.4f}, MAE_log={mae_log:.4f}")
            rows[j]["RMSE_log"] = rmse_log; rows[j]["MAE_log"] = mae_log

    tag = run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(FORECAST_OUT, f"metrics_summary_{tag}.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics CSV to {metrics_csv}")

    # Bar chart
    x = np.arange(len(horizons)); width = 0.2
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x - 1.5*width, metrics_df["RMSE"], width, label="RMSE")
    ax.bar(x - 0.5*width, metrics_df["MAE"],  width, label="MAE")
    ax.bar(x + 0.5*width, metrics_df["sMAPE_percent"], width, label="sMAPE (%)")
    ax.bar(x + 1.5*width, metrics_df["R2"], width, label="R²")
    ax.set_xticks(x); ax.set_xticklabels([f"t+{h}" for h in metrics_df["horizon"]])
    ax.set_title("Evaluation Metrics by Horizon")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    metrics_png = os.path.join(FORECAST_OUT, f"metrics_summary_{tag}.png")
    plt.savefig(metrics_png, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved metrics plot to {metrics_png}")

    return metrics_df

@torch.no_grad()
def predict_for_dates(
    models: ModelOrEnsemble,
    train_index_min: pd.Timestamp,
    base_columns: List[str],
    dates: List[pd.Timestamp],
    horizons: List[int],
    *,
    device=DEVICE,
    ft=None,
    log_target: bool = False,
    expm1_cap_log: float | None = None,
    scale_calibration: Optional[np.ndarray] = None,
) -> pd.DataFrame:

    dates_idx = pd.to_datetime(pd.Index(dates)).normalize()
    f = pd.DataFrame(index=dates_idx)
    f = add_time_features(f, dates_idx)
    f["days_since_start"] = (dates_idx - train_index_min).days

    for c in base_columns:
        if c not in f.columns: f[c] = 0
    f = f.reindex(columns=base_columns, fill_value=0)

    if ft is not None:
        f = ft.transform(f)

    dummy_y = pd.Series(np.zeros(len(f), dtype=np.float32), index=f.index)
    ds = DailyDataset(f, dummy_y, horizons)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    outs = []
    for cat, cont, _, _ in loader:
        pred = _predict_batch_reg(models, cat.to(DEVICE), cont.to(DEVICE))
        outs.append(pred)
    P = np.vstack(outs)  # [T,H]

    if log_target:
        cap_log = float(expm1_cap_log) if expm1_cap_log is not None else 15.0
        P = np.expm1(np.clip(P, None, cap_log))

    if scale_calibration is not None:
        P = P * scale_calibration.reshape(1, -1)

    frames = []
    for j, h in enumerate(horizons):
        target_dates = dates_idx + pd.to_timedelta(h, unit="D")
        frames.append(
            pd.DataFrame({"date": target_dates, f"predicted_tickets_t+{h}": P[:, j].astype(float)}).set_index("date")
        )
    return pd.concat(frames, axis=1)

def allocate_to_grid(pred_df: pd.DataFrame, spatial_ref: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
    if spatial_ref is None or spatial_ref.empty:
        raise ValueError("spatial_ref empty; cannot allocate to grid.")
    out = {}
    for date in pred_df.index:
        dow = pd.Timestamp(date).dayofweek
        month = pd.Timestamp(date).month
        pri = spatial_ref[(spatial_ref["day_of_week"]==dow) & (spatial_ref["month"]==month)]
        if pri.empty:
            pri = spatial_ref.groupby(["lat_grid", "lon_grid"], as_index=False)["weight"].mean()
        pri = pri.copy()
        total_pred = float(np.nan_to_num(pred_df.loc[date].filter(like="predicted_tickets").sum(), nan=0.0))
        pri["allocated"] = pri["weight"] * max(0.0, total_pred)
        out[pd.Timestamp(date)] = pri[["lat_grid", "lon_grid", "allocated"]].reset_index(drop=True)
    return out

def run_forecast(args):
    set_all_seeds(RANDOM_STATE)
    df = pd.read_csv(FORECAST_DATA_PATH, low_memory=False)
    X_base, y_base, spatial_ref = build_daily_matrix(df, use_recency=args.use_recency)

    # split
    X_tr, X_te, y_tr, y_te = split_time(X_base, y_base, TEST_FRACTION)
    n_tr = len(X_tr); n_val = max(1, int(np.floor(VAL_FRACTION * n_tr)))
    X_train, X_val = X_tr.iloc[:-n_val], X_tr.iloc[-n_val:]
    y_train, y_val = y_tr.iloc[:-n_val], y_tr.iloc[-n_val:]

    # transforms
    ft = FeatureTransformer(); ft.fit(X_train)
    X_train_f = ft.transform(X_train); X_val_f = ft.transform(X_val); X_te_f = ft.transform(X_te)

    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    max_train = float(max(1, y_train.max()))
    cap_log = float(np.log1p(max_train) + 2.0)

    # Train ensemble
    models: List[nn.Module] = []
    for s in range(args.seeds):
        seed = RANDOM_STATE + s*13
        m = train_one_model_forecast(
            X_train_f, X_val_f, y_train, y_val,
            HORIZONS, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            device=DEVICE, log_target=args.log_target, run_tag=run_tag,
            huber_beta=args.huber_beta, horizon_weights=args.horizon_weights, seed=seed
        )
        models.append(m)
    ensemble: ModelOrEnsemble = models if len(models) > 1 else models[0]

    # ----- optional scale-only calibration on validation -----
    scale_cal = None
    if args.calibrate_scale:
        # build validation preds in raw space
        ds_va = DailyDataset(X_val_f, y_val, HORIZONS)
        va_loader = DataLoader(ds_va, batch_size=512, shuffle=False)
        Yt_list, Yp_list, Mk_list = [], [], []
        with torch.no_grad():
            for cat, cont, y, m in va_loader:
                pred = _predict_batch_reg(ensemble, cat.to(DEVICE), cont.to(DEVICE))
                Yt_list.append(y.numpy()); Yp_list.append(pred); Mk_list.append(m.numpy())
        Yt = np.concatenate(Yt_list); Yp = np.concatenate(Yp_list); Mk = np.concatenate(Mk_list)
        if args.log_target:
            Yp_raw = np.expm1(np.clip(Yp, None, cap_log))
        else:
            Yp_raw = Yp
        scale_cal = fit_scale_only_calibration(Yp_raw, Yt, Mk)
        print(f"Calibration fitted (scale-only). a={np.round(scale_cal,3)}")

    # Evaluate
    _ = evaluate_forecast(
        ensemble, X_te_f, y_te, HORIZONS, device=DEVICE, log_target=args.log_target, run_tag=run_tag,
        expm1_cap_log=cap_log if args.log_target else None, scale_calibration=scale_cal
    )

    # Plots + overlays
    if args.plot:
        for h in HORIZONS:
            y_h = y_base.shift(-h); mask = y_h.notna()
            X_h, y_h = X_base.loc[mask], y_h.loc[mask].astype(int)
            _X_trh, X_teh, _y_trh, y_teh = split_time(X_h, y_h, TEST_FRACTION)
            X_teh_f = ft.transform(X_teh)

            with torch.no_grad():
                ds = DailyDataset(X_teh_f, y_teh, [h])
                loader = DataLoader(ds, batch_size=512, shuffle=False)
                preds = []
                for cat, cont, _, _ in loader:
                    pred = _predict_batch_reg(ensemble, cat.to(DEVICE), cont.to(DEVICE))[:, 0]
                    preds.append(pred)
                y_pred = np.concatenate(preds)
            if args.log_target:
                y_pred = np.expm1(np.clip(y_pred, None, cap_log))
            if scale_cal is not None:
                y_pred = y_pred * scale_cal[HORIZONS.index(h)]

            overlay = pd.DataFrame({"date": y_teh.index, "actual": y_teh.values, "predicted": y_pred}).set_index("date")
            overlay_csv = os.path.join(FORECAST_OUT, f"overlay_t+{h}_{run_tag}.csv")
            overlay.to_csv(overlay_csv)
            print(f"Saved overlay CSV to {overlay_csv}")

            plt.figure(figsize=(12, 5))
            for day in y_teh.index:
                if day.weekday() >= 5: plt.axvspan(day, day + pd.Timedelta(days=1), alpha=0.12)
            for day in y_teh.index:
                if is_holiday(day): plt.axvline(day, linestyle=":", alpha=0.7)
            plt.plot(y_teh.index, y_teh.values, label="Actual")
            plt.plot(y_teh.index, y_pred, label="Predicted", linestyle="--")
            plt.title(f"Pred vs Actual – TabTransformer (t+{h})")
            plt.xlabel("Date"); plt.ylabel("Violations"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            save_path = os.path.join(FORECAST_OUT, f"pred_vs_actual_t+{h}_{run_tag}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
            print(f"Saved t+{h} test plot to {save_path}")

    # forecast CSV + spatial sample
    if args.forecast_days and args.forecast_days > 0:
        start = X_base.index.max() + pd.Timedelta(days=1)
        fut_idx = pd.date_range(start=start, periods=args.forecast_days, freq="D")
        preds = predict_for_dates(
            ensemble, train_index_min=X_base.index.min(), base_columns=list(X_base.columns), dates=fut_idx,
            horizons=HORIZONS, device=DEVICE, ft=ft, log_target=args.log_target,
            expm1_cap_log=cap_log if args.log_target else None, scale_calibration=scale_cal
        )
        forecast_csv = os.path.join(FORECAST_OUT, f"forecast_{run_tag}.csv")
        preds.round(2).to_csv(forecast_csv)
        print(f"Saved forecast CSV to {forecast_csv}")

        try:
            alloc = allocate_to_grid(preds, spatial_ref)
            any_date = sorted(alloc.keys())[0]
            sample = alloc[any_date].sort_values("allocated", ascending=False).head(20)
            alloc_csv = os.path.join(FORECAST_OUT, f"allocation_{any_date.date()}_{run_tag}.csv")
            sample.to_csv(alloc_csv, index=False)
            print(f"Saved sample allocation CSV to {alloc_csv}")
        except Exception as e:
            print("Spatial allocation skipped:", e)


# ==============================================================
#            GEO POINT PROBABILITY CLASSIFIER
# ==============================================================

@dataclass
class BBox:
    lat_min: float = 38.78
    lat_max: float = 39.03
    lon_min: float = -77.12
    lon_max: float = -76.90

def load_parking_violation_data(data_dir: str) -> pd.DataFrame:
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")
    frames = []
    for p in csvs:
        try:
            frames.append(pd.read_csv(p, low_memory=False))
            print(f"loaded {os.path.basename(p)}")
        except Exception as e:
            print(f"skipped {p}: {e}")
    return pd.concat(frames, ignore_index=True)

def load_clean_df_geo(data_dir: str, grid_round: int = 3, bbox: Optional[BBox] = BBox()) -> pd.DataFrame:
    df = load_parking_violation_data(data_dir)

    # find datetime col
    dt_col = None
    for c in df.columns:
        cl = c.lower()
        if "issue" in cl and ("datetime" in cl or ("date" in cl and "time" in cl)):
            dt_col = c; break
    if dt_col is None:
        for c in df.columns:
            if "date" in c.lower():
                dt_col = c; break
    if dt_col is None:
        raise ValueError("Could not infer datetime column.")

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.rename(columns={dt_col: "issue_datetime"})

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Expected 'latitude' and 'longitude' columns.")

    df = df.dropna(subset=["issue_datetime", "latitude", "longitude"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    if bbox is not None:
        df = df[
            (df["latitude"].between(bbox.lat_min, bbox.lat_max)) &
            (df["longitude"].between(bbox.lon_min, bbox.lon_max))
        ].copy()

    df["hour"] = df["issue_datetime"].dt.hour.astype(int)
    df["day_of_week"] = df["issue_datetime"].dt.dayofweek.astype(int)
    df["month"] = df["issue_datetime"].dt.month.astype(int)
    df["date"] = df["issue_datetime"].dt.date

    df["lat_grid"] = df["latitude"].round(grid_round)
    df["lon_grid"] = df["longitude"].round(grid_round)
    df["hour_bucket"] = df["hour"]

    return df[["issue_datetime","latitude","longitude","lat_grid","lon_grid","hour_bucket","day_of_week","month","date"]].copy()

def build_positive_negative_samples(df: pd.DataFrame, max_neg_per_pos: float = 1.0) -> pd.DataFrame:
    pos = (df.groupby(["date","lat_grid","lon_grid","hour_bucket"]).size()
             .rename("count").reset_index())
    pos["label"] = 1

    # For each date, visited grids that day
    grids_by_date = (
        df.groupby("date")[["lat_grid","lon_grid"]]
          .apply(lambda g: g.drop_duplicates().reset_index(drop=True))
          .reset_index(name="grid_df")
    )

    rows = []
    for _, row in grids_by_date.iterrows():
        date = row["date"]; grid_df = row["grid_df"]
        for _, gg in grid_df.iterrows():
            for h in range(24):
                rows.append((date, float(gg["lat_grid"]), float(gg["lon_grid"]), h))
    full = pd.DataFrame(rows, columns=["date","lat_grid","lon_grid","hour_bucket"])
    full = full.merge(pos[["date","lat_grid","lon_grid","hour_bucket","label"]],
                      how="left", on=["date","lat_grid","lon_grid","hour_bucket"])
    full["label"] = full["label"].fillna(0).astype(int)

    num_pos = int((full["label"] == 1).sum())
    if num_pos == 0:
        raise ValueError("No positive samples found.")
    neg = full[full["label"] == 0]
    target_neg = int(min(len(neg), num_pos * max_neg_per_pos))
    if target_neg < len(neg):
        neg = neg.sample(n=target_neg, random_state=RANDOM_STATE)

    data = pd.concat([full[full["label"] == 1], neg], axis=0).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # representative lat/lon for positives; negatives use grid center
    rep = df.groupby(["date","lat_grid","lon_grid","hour_bucket"]).agg(
        latitude=("latitude","median"),
        longitude=("longitude","median"),
        day_of_week=("day_of_week","median"),
        month=("month","median"),
    ).reset_index()
    data = data.merge(rep, on=["date","lat_grid","lon_grid","hour_bucket"], how="left")
    data["latitude"]  = data["latitude"].fillna(data["lat_grid"])
    data["longitude"] = data["longitude"].fillna(data["lon_grid"])

    miss = data["day_of_week"].isna()
    if miss.any():
        dts = pd.to_datetime(data.loc[miss, "date"])
        data.loc[miss, "day_of_week"] = dts.dt.dayofweek.values
        data.loc[miss, "month"] = dts.dt.month.values

    data["day_of_week"] = data["day_of_week"].astype(int)
    data["month"] = data["month"].astype(int)

    return data

def fit_feature_meta(df: pd.DataFrame) -> dict:
    meta = {}
    meta["lat_min"], meta["lat_max"] = df["latitude"].min(), df["latitude"].max()
    meta["lon_min"], meta["lon_max"] = df["longitude"].min(), df["longitude"].max()
    meta["hour_cardinality"] = 24; meta["dow_cardinality"] = 7; meta["month_cardinality"] = 12
    # grid map
    grid_pairs = df[["lat_grid","lon_grid"]].drop_duplicates().reset_index(drop=True)
    grid_pairs["grid_id"] = np.arange(len(grid_pairs))
    meta["grid_lat_lon"] = grid_pairs.to_dict(orient="list")
    meta["month_mode"] = int(df["month"].mode().iloc[0]) if len(df) else 6
    return meta

def save_feature_meta(meta: dict, path: str):
    np.savez(path, **meta)

def load_feature_meta(path: str) -> dict:
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k].item() if npz[k].shape == () else npz[k] for k in npz.files}

def encode_features(df: pd.DataFrame, meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    grid_df = pd.DataFrame(meta["grid_lat_lon"])
    df = df.merge(grid_df, on=["lat_grid","lon_grid"], how="left")
    if df["grid_id"].isna().any():
        # simple nearest in rounded space
        for idx in df.index[df["grid_id"].isna()]:
            lat_g = df.at[idx, "lat_grid"]; lon_g = df.at[idx, "lon_grid"]
            diff = (grid_df["lat_grid"] - lat_g)**2 + (grid_df["lon_grid"] - lon_g)**2
            j = diff.idxmin(); df.at[idx, "grid_id"] = int(grid_df.loc[j, "grid_id"])
    df["grid_id"] = df["grid_id"].astype(int)

    # normalize lat/lon
    lat = df["latitude"].to_numpy(); lon = df["longitude"].to_numpy()
    lat_norm = (lat - meta["lat_min"]) / max(1e-6, (meta["lat_max"] - meta["lat_min"])) * 2 - 1
    lon_norm = (lon - meta["lon_min"]) / max(1e-6, (meta["lon_max"] - meta["lon_min"])) * 2 - 1

    hour = df["hour_bucket"].astype(int).to_numpy()
    dow  = df["day_of_week"].astype(int).to_numpy()
    month= df["month"].astype(int).to_numpy()

    hour_sin = np.sin(2 * np.pi * hour / 24.0);  hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin  = np.sin(2 * np.pi * dow  / 7.0 );  dow_cos  = np.cos(2 * np.pi * dow  / 7.0 )
    month_sin= np.sin(2 * np.pi * (month-1) / 12.0); month_cos= np.cos(2 * np.pi * (month-1) / 12.0)

    cont = np.stack([lat_norm, lon_norm, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos], axis=1).astype(np.float32)
    cat  = np.stack([hour, dow, month-1, df["grid_id"].to_numpy().astype(int)], axis=1)
    y    = df["label"].to_numpy().astype(np.float32)

    grid_card = int(cat[:,3].max()) + 1
    return cat, cont, y, grid_card

class TicketDataset(Dataset):
    def __init__(self, cat_tokens, cont_feats, labels):
        self.cat = torch.tensor(cat_tokens, dtype=torch.long)
        self.cont = torch.tensor(cont_feats, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.cat[idx], self.cont[idx], self.y[idx]

class TabTransformerCls(nn.Module):
    def __init__(self, hour_card=24, dow_card=7, month_card=12, grid_card=10000,
                 d_token=48, n_heads=4, n_layers=2, cont_dim=8, mlp_hidden=96, dropout=0.1):
        super().__init__()
        self.hour_emb  = nn.Embedding(hour_card, d_token)
        self.dow_emb   = nn.Embedding(dow_card, d_token)
        self.month_emb = nn.Embedding(month_card, d_token)
        self.grid_emb  = nn.Embedding(grid_card, d_token)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cont_proj = nn.Sequential(nn.Linear(cont_dim, d_token), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(d_token*5, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )
    def forward(self, cat_tokens, cont_feats):
        h = self.hour_emb(cat_tokens[:,0]); d = self.dow_emb(cat_tokens[:,1]); m = self.month_emb(cat_tokens[:,2]); g = self.grid_emb(cat_tokens[:,3])
        tokens = torch.stack([h,d,m,g], dim=1)    
        z = self.encoder(tokens).mean(dim=1)  
        c = self.cont_proj(cont_feats)    
        x = torch.cat([z, h, d, m, c], dim=1)    
        logits = self.head(x).squeeze(1)
        return logits

def geo_prepare_loaders(data_dir: str, max_neg_per_pos=1.0, batch_size=1024, test_size=0.2, val_size=0.1):
    df = load_clean_df_geo(data_dir)
    data = build_positive_negative_samples(df, max_neg_per_pos=max_neg_per_pos)
    meta = fit_feature_meta(data)
    cat, cont, labels, grid_card = encode_features(data, meta)

    Xc_tr, Xc_tmp, Xn_tr, Xn_tmp, y_tr, y_tmp = train_test_split(
        cat, cont, labels, test_size=test_size, random_state=RANDOM_STATE, stratify=labels
    )
    val_ratio = val_size / (1 - test_size)
    Xc_va, Xc_te, Xn_va, Xn_te, y_va, y_te = train_test_split(
        Xc_tmp, Xn_tmp, y_tmp, test_size=1 - val_ratio, random_state=RANDOM_STATE, stratify=y_tmp
    )

    train_ds = TicketDataset(Xc_tr, Xn_tr, y_tr)
    val_ds   = TicketDataset(Xc_va, Xn_va, y_va)
    test_ds  = TicketDataset(Xc_te, Xn_te, y_te)

    loaders = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    )
    return loaders, meta, grid_card

def geo_train(args):
    (train_loader, val_loader, test_loader), meta, grid_card = geo_prepare_loaders(
        args.data_dir, max_neg_per_pos=args.max_neg_per_pos, batch_size=args.batch_size
    )
    model = TabTransformerCls(grid_card=grid_card, d_token=48, n_heads=4, n_layers=2, cont_dim=8, mlp_hidden=96, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best_auc = -1.0; best_state = None
    for epoch in range(1, args.epochs+1):
        model.train(); total = 0.0
        for cat, cont, y in train_loader:
            cat, cont, y = cat.to(DEVICE), cont.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); logits = model(cat, cont); loss = crit(logits, y)
            loss.backward(); opt.step(); total += loss.item() * len(y)
        avg_loss = total / len(train_loader.dataset)

        # val
        model.eval(); y_true=[]; y_prob=[]
        with torch.no_grad():
            for cat, cont, y in val_loader:
                p = torch.sigmoid(model(cat.to(DEVICE), cont.to(DEVICE))).cpu().numpy()
                y_prob.append(p); y_true.append(y.numpy())
        y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = float("nan")
        ap = float(average_precision_score(y_true, y_prob))
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  val_auc={auc:.4f}  val_ap={ap:.4f}")
        if auc > best_auc:
            best_auc = auc; best_state = model.state_dict()

    if best_state is not None: model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(GEO_OUT, "geo_tabtransformer.pt"))
    save_feature_meta(meta, os.path.join(GEO_OUT, "geo_feature_meta.npz"))
    print("Saved geo model & encoder to", GEO_OUT)

    # quick test metrics
    model.eval(); y_true=[]; y_prob=[]
    with torch.no_grad():
        for cat, cont, y in test_loader:
            p = torch.sigmoid(model(cat.to(DEVICE), cont.to(DEVICE))).cpu().numpy()
            y_prob.append(p); y_true.append(y.numpy())
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    auc = roc_auc_score(y_true, y_prob); ap = average_precision_score(y_true, y_prob)
    print("TEST AUC:", round(auc,4), "TEST AP:", round(ap,4))
    print(classification_report((y_prob>0.5).astype(int), y_true.astype(int)))

def geo_eval(args):
    ( _ , _ , test_loader), meta, grid_card = geo_prepare_loaders(args.data_dir)
    model = TabTransformerCls(grid_card=grid_card, d_token=48, n_heads=4, n_layers=2, cont_dim=8, mlp_hidden=96, dropout=0.1).to(DEVICE)
    state = torch.load(os.path.join(GEO_OUT, "geo_tabtransformer.pt"), map_location=DEVICE)
    model.load_state_dict(state); model.eval()
    y_true=[]; y_prob=[]
    with torch.no_grad():
        for cat, cont, y in test_loader:
            p = torch.sigmoid(model(cat.to(DEVICE), cont.to(DEVICE))).cpu().numpy()
            y_prob.append(p); y_true.append(y.numpy())
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    auc = roc_auc_score(y_true, y_prob); ap = average_precision_score(y_true, y_prob)
    print("TEST AUC:", round(auc,4), "TEST AP:", round(ap,4))
    print(classification_report((y_prob>0.5).astype(int), y_true.astype(int)))

def _nearest_grid(lat, lon, grid_df: pd.DataFrame):
    lat_g = round(lat, 3); lon_g = round(lon, 3)
    match = grid_df[(grid_df["lat_grid"]==lat_g) & (grid_df["lon_grid"]==lon_g)]
    if len(match) == 0:
        diff = (grid_df["lat_grid"] - lat_g)**2 + (grid_df["lon_grid"] - lon_g)**2
        idx = diff.idxmin()
        return int(grid_df.loc[idx, "grid_id"]), float(grid_df.loc[idx, "lat_grid"]), float(grid_df.loc[idx, "lon_grid"])
    else:
        return int(match.iloc[0]["grid_id"]), lat_g, lon_g

def geo_predict(args):
    meta = load_feature_meta(os.path.join(GEO_OUT, "geo_feature_meta.npz"))
    grid_df = pd.DataFrame(meta["grid_lat_lon"])
    grid_card = int(max(grid_df["grid_id"])) + 1
    model = TabTransformerCls(grid_card=grid_card, d_token=48, n_heads=4, n_layers=2, cont_dim=8, mlp_hidden=96, dropout=0.1).to(DEVICE)
    state = torch.load(os.path.join(GEO_OUT, "geo_tabtransformer.pt"), map_location=DEVICE)
    model.load_state_dict(state); model.eval()

    if args.when is not None:
        when = pd.to_datetime(args.when, errors="coerce")
        if pd.isna(when): raise ValueError("Invalid --when datetime")
        hour = int(when.hour); dow = int(when.dayofweek); month = int(when.month)
    else:
        if args.hour is None or args.dow is None:
            raise ValueError("Provide --when OR (--hour and --dow [and optional --month])")
        hour = int(args.hour); dow = int(args.dow); month = int(args.month) if args.month else int(meta.get("month_mode", 6))

    grid_id, _, _ = _nearest_grid(args.lat, args.lon, grid_df)

    lat_norm = (args.lat - meta["lat_min"]) / max(1e-6, (meta["lat_max"] - meta["lat_min"])) * 2 - 1
    lon_norm = (args.lon - meta["lon_min"]) / max(1e-6, (meta["lon_max"] - meta["lon_min"])) * 2 - 1

    hour_sin = math.sin(2 * math.pi * hour / 24.0); hour_cos = math.cos(2 * math.pi * hour / 24.0)
    dow_sin  = math.sin(2 * math.pi * dow / 7.0);    dow_cos  = math.cos(2 * math.pi * dow / 7.0)
    month_sin= math.sin(2 * math.pi * (month - 1) / 12.0); month_cos= math.cos(2 * math.pi * (month - 1) / 12.0)

    cont = torch.tensor([[lat_norm, lon_norm, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos]], dtype=torch.float32).to(DEVICE)
    cat  = torch.tensor([[hour, dow, month-1, grid_id]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(cat, cont)).item()
    print(f"Predicted probability: {prob*100:.2f}%")

# ==============================================================
#                              CLI
# ==============================================================

def main():
    p = argparse.ArgumentParser(description="One-stop TabTransformer: Forecast + Geo probability")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Forecast
    pf = sub.add_parser("forecast", help="Train/evaluate daily multi-horizon forecast")
    pf.add_argument("--epochs", type=int, default=200)
    pf.add_argument("--batch_size", type=int, default=256)
    pf.add_argument("--lr", type=float, default=1e-3)
    pf.add_argument("--plot", action="store_true")
    pf.add_argument("--forecast_days", type=int, default=0)
    pf.add_argument("--log_target", action="store_true")
    pf.add_argument("--seeds", type=int, default=1)
    pf.add_argument("--huber_beta", type=float, default=HUBER_BETA_D)
    pf.add_argument("--horizon_weights", nargs=len(HORIZONS), type=float, default=HORIZON_WEIGHTS_DEFAULT)
    pf.add_argument("--use_recency", action="store_true")
    pf.add_argument("--calibrate_scale", action="store_true", help="Fit per-horizon scale on validation to reduce bias")

    # Geo
    pgt = sub.add_parser("geo-train", help="Train geo grid probability model")
    pgt.add_argument("--data_dir", type=str, default="../CleanData")
    pgt.add_argument("--epochs", type=int, default=10)
    pgt.add_argument("--batch_size", type=int, default=1024)
    pgt.add_argument("--lr", type=float, default=1e-3)
    pgt.add_argument("--max_neg_per_pos", type=float, default=1.0)

    pge = sub.add_parser("geo-eval", help="Evaluate geo model")
    pge.add_argument("--data_dir", type=str, default="../CleanData")

    pgp = sub.add_parser("geo-predict", help="Predict P(ticket) for lat/lon/time")
    pgp.add_argument("--lat", type=float, required=True)
    pgp.add_argument("--lon", type=float, required=True)
    pgp.add_argument("--when", type=str, default=None)
    pgp.add_argument("--hour", type=str, default=None)
    pgp.add_argument("--dow", type=str, default=None)
    pgp.add_argument("--month", type=str, default=None)

    args = p.parse_args()

    if args.cmd == "forecast":
        run_forecast(args)
    elif args.cmd == "geo-train":
        geo_train(args)
    elif args.cmd == "geo-eval":
        geo_eval(args)
    elif args.cmd == "geo-predict":
        geo_predict(args)

if __name__ == "__main__":
    main()
