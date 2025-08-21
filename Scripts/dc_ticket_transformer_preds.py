"""
DC Parking Tickets – TabTransformer Regression (DAILY, direct multi-horizon h={1,3,7})
- Single model predicts residuals for +1d, +3d, +7d using only info available at time t
- Target_h(t) = y_log(t+h) - roll7_log(t). Prediction: y_hat(t+h) = expm1(roll7_log(t) + res_hat_h(t))
- Chronological split; early stopping on WAPE(avg over horizons); grad clipping; LR scheduler
- Outputs metrics + plots per horizon, plus overall (micro + macro) metrics saved to metrics.json
"""
import os, json, math, random, argparse
from datetime import datetime
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# -----------------------------
# Config
# -----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

DATA_CSV = "../CleanData/prepared_data.csv"
HORIZONS = [1, 3, 7]

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")

# -----------------------------
# utils
# -----------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def make_run_dir():
    ensure_dir("artifacts")
    run_dir = os.path.join("artifacts", datetime.now().strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_dir)
    latest = os.path.join("artifacts", "latest")
    try:
        if os.path.islink(latest): os.unlink(latest)
        elif os.path.exists(latest): os.remove(latest)
        os.symlink(os.path.abspath(run_dir), latest, target_is_directory=True)
    except Exception:
        with open(os.path.join("artifacts","LATEST_PATH.txt"), "w") as f:
            f.write(os.path.abspath(run_dir))
    return run_dir

def smape(y, yhat, eps=1e-6):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(100.0*np.mean(np.abs(y-yhat)/((np.abs(y)+np.abs(yhat))/2.0+eps)))

def wape(y, yhat, eps=1e-6):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(100.0*np.sum(np.abs(y-yhat))/(np.sum(np.abs(y))+eps))

def rmse(y, yhat): return math.sqrt(mean_squared_error(y, yhat))

# -----------------------------
# Data prep
# -----------------------------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)

    dt = pd.to_datetime(df["issue_datetime"], errors="coerce", utc=True)

    dt = dt.dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["issue_datetime"] = dt
    df = df.dropna(subset=["issue_datetime"]).copy()
    return df


def make_daily(df: pd.DataFrame, holiday_dates: List[pd.Timestamp] = None) -> pd.DataFrame:
    df["_d"] = df["issue_datetime"].dt.floor("D")
    daily = df.groupby("_d").size().rename("tickets").to_frame().sort_index()
    idx = daily.index
    feat = pd.DataFrame(index=idx)
    feat["tickets"] = daily["tickets"].astype(float)
    feat["y_log"]   = np.log1p(feat["tickets"].values)

    feat["dow"]   = idx.dayofweek
    feat["mon"]   = idx.month
    feat["doy"]   = idx.dayofyear
    feat["is_weekend"] = (feat["dow"] >= 5).astype(int)

    feat["is_holiday"] = 0
    if holiday_dates:
        hd = pd.Series(1, index=pd.to_datetime(pd.Index(holiday_dates)).floor("D"))
        feat.loc[feat.index.isin(hd.index), "is_holiday"] = 1

    # Cyclic encodings
    feat["dow_sin"] = np.sin(2*np.pi*feat["dow"]/7.0)
    feat["dow_cos"] = np.cos(2*np.pi*feat["dow"]/7.0)
    feat["mon_sin"] = np.sin(2*np.pi*(feat["mon"]-1)/12.0)
    feat["mon_cos"] = np.cos(2*np.pi*(feat["mon"]-1)/12.0)
    feat["doy_sin"] = np.sin(2*np.pi*(feat["doy"]-1)/366.0)
    feat["doy_cos"] = np.cos(2*np.pi*(feat["doy"]-1)/366.0)

    # Past-only logs
    ylog = feat["y_log"]
    feat["lag1_log"]   = ylog.shift(1)
    feat["lag7_log"]   = ylog.shift(7)
    feat["roll7_log"]  = ylog.rolling(7, min_periods=1).mean().shift(1)   # up to t-1
    feat["roll14_log"] = ylog.rolling(14, min_periods=1).mean().shift(1)
    feat["sdiff7_log"] = feat["lag1_log"] - feat["roll7_log"]

    # Safe starts
    for col in ["lag1_log","lag7_log","roll7_log","roll14_log","sdiff7_log"]:
        m = float(feat[col].mean(skipna=True))
        feat[col] = feat[col].fillna(m)

    feat["ts"] = idx
    return feat.reset_index(drop=True)


def standardize_hist(feat: pd.DataFrame) -> Dict[str, float]:
    def nzstd(a):
        s = float(np.std(a, ddof=0)); return s if s>1e-9 else 1.0
    meta = dict(
        lag1_mean=float(feat["lag1_log"].mean()),   lag1_std=nzstd(feat["lag1_log"]),
        lag7_mean=float(feat["lag7_log"].mean()),   lag7_std=nzstd(feat["lag7_log"]),
        roll7_mean=float(feat["roll7_log"].mean()), roll7_std=nzstd(feat["roll7_log"]),
        roll14_mean=float(feat["roll14_log"].mean()), roll14_std=nzstd(feat["roll14_log"]),
        sdev_mean=float(feat["sdiff7_log"].mean()), sdev_std=nzstd(feat["sdiff7_log"]),
        cont_names=["dow_sin","dow_cos","mon_sin","mon_cos","doy_sin","doy_cos",
                    "lag1_z","lag7_z","roll7_z","roll14_z","sdiff7_z"]
    )
    return meta

def encode(feat: pd.DataFrame, meta: dict) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    # categorical tokens
    cat = np.stack([
        feat["dow"].to_numpy().astype(int),
        (feat["mon"].to_numpy().astype(int)-1),
        feat["is_weekend"].to_numpy().astype(int),
        feat["is_holiday"].to_numpy().astype(int),
    ], axis=1)

    # standardize past-only continuous
    lag1_z  = (feat["lag1_log"]  - meta["lag1_mean"])   / meta["lag1_std"]
    lag7_z  = (feat["lag7_log"]  - meta["lag7_mean"])   / meta["lag7_std"]
    roll7_z = (feat["roll7_log"] - meta["roll7_mean"])  / meta["roll7_std"]
    roll14_z= (feat["roll14_log"]- meta["roll14_mean"]) / meta["roll14_std"]
    sdev7_z = (feat["sdiff7_log"]- meta["sdev_mean"])   / meta["sdev_std"]

    cont = np.stack([
        feat["dow_sin"].astype(np.float32),
        feat["dow_cos"].astype(np.float32),
        feat["mon_sin"].astype(np.float32),
        feat["mon_cos"].astype(np.float32),
        feat["doy_sin"].astype(np.float32),
        feat["doy_cos"].astype(np.float32),
        lag1_z.astype(np.float32), lag7_z.astype(np.float32),
        roll7_z.astype(np.float32), roll14_z.astype(np.float32), sdev7_z.astype(np.float32),
    ], axis=1)

    y_log = feat["y_log"].to_numpy().astype(np.float32)
    ts    = pd.to_datetime(feat["ts"]).to_numpy()
    roll7 = feat["roll7_log"].to_numpy().astype(np.float32)
    return cat, cont, y_log, ts, roll7

def build_direct_h_targets(y_log: np.ndarray, roll7_log_t: np.ndarray, horizons=HORIZONS):
    """
    For each t, and each horizon h, target is y_log(t+h) - roll7_log(t).
    Return: Y_targets shape [N, H]; mask to drop rows where any target missing.
    """
    N = len(y_log); H = len(horizons)
    Y = np.full((N, H), np.nan, dtype=np.float32)
    for j, h in enumerate(horizons):
        y_future = np.r_[y_log[h:], np.full(h, np.nan, dtype=np.float32)]
        Y[:, j] = y_future - roll7_log_t
    # keep only rows where all horizons are available
    valid = np.all(~np.isnan(Y), axis=1)
    return Y[valid], valid

def chrono_split_direct(cat, cont, roll7, y_log, ts, Y_targets,
                        train_frac=0.7, val_frac=0.15):
    # all arrays already trimmed to valid positions
    n = len(Y_targets)
    tr = int(train_frac*n); va = int((train_frac+val_frac)*n)
    split = lambda a: (a[:tr], a[tr:va], a[va:])
    Xc_tr,Xc_va,Xc_te = split(cat); Xn_tr,Xn_va,Xn_te = split(cont)
    ro_tr,ro_va,ro_te = split(roll7); y_tr,y_va,y_te   = split(y_log)
    ts_tr,ts_va,ts_te = split(ts);    T_tr,T_va,T_te   = split(Y_targets)
    return (Xc_tr,Xn_tr,ro_tr,y_tr,ts_tr,T_tr), (Xc_va,Xn_va,ro_va,y_va,ts_va,T_va), (Xc_te,Xn_te,ro_te,y_te,ts_te,T_te)

class TicketHorizonDataset(Dataset):
    def __init__(self, cat, cont, roll7, y_log, ts, targets):
        self.cat=torch.tensor(cat, dtype=torch.long)
        self.cont=torch.tensor(cont, dtype=torch.float32)
        self.roll7=torch.tensor(roll7, dtype=torch.float32)
        self.ylog=torch.tensor(y_log, dtype=torch.float32)
        self.ts = np.array(ts)
        self.targets=torch.tensor(targets, dtype=torch.float32)  # [B, H]
    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        return self.cat[i], self.cont[i], self.roll7[i], self.ylog[i], self.targets[i]

# -----------------------------
# Model (multi-head for H horizons)
# -----------------------------
class TabTransformerMultiH(nn.Module):
    def __init__(self, cardinals: Dict[str,int], cont_dim: int, d_token=64,
                 n_heads=4, n_layers=2, dropout=0.25, n_horizons=len(HORIZONS)):
        super().__init__()
        self.dow_emb   = nn.Embedding(cardinals["dow"], d_token)
        self.mon_emb   = nn.Embedding(cardinals["mon"], d_token)
        self.wkend_emb = nn.Embedding(cardinals["wk"],  d_token)
        self.hol_emb   = nn.Embedding(cardinals["hol"], d_token)

        self.cls = nn.Parameter(torch.zeros(1,1,d_token))
        nn.init.trunc_normal_(self.cls, std=0.02)

        enc = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token*4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)

        self.cont_proj = nn.Sequential(
            nn.Linear(cont_dim, d_token), nn.GELU(), nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.Linear(d_token*2, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_horizons)
        )

    def forward(self, cat_tokens, cont_feats):
        d = self.dow_emb(cat_tokens[:,0])
        m = self.mon_emb(cat_tokens[:,1])
        w = self.wkend_emb(cat_tokens[:,2])
        h = self.hol_emb(cat_tokens[:,3])
        tok = torch.stack([d,m,w,h], dim=1) 
        cls = self.cls.expand(tok.size(0), -1, -1)
        z   = self.encoder(torch.cat([cls, tok], dim=1))[:,0] 
        c   = self.cont_proj(cont_feats)
        return self.head(torch.cat([z, c], dim=1))       


# -----------------------------
# Training / evaluation
# -----------------------------
def _encode_with_meta(feat: pd.DataFrame, meta: dict):
    # categorical tokens
    cat = np.stack([
        feat["dow"].to_numpy().astype(int),
        (feat["mon"].to_numpy().astype(int)-1),
        feat["is_weekend"].to_numpy().astype(int),
        feat["is_holiday"].to_numpy().astype(int),
    ], axis=1)

    # standardize past-only using TRAIN stats
    lag1_z  = (feat["lag1_log"]  - meta["lag1_mean"])   / meta["lag1_std"]
    lag7_z  = (feat["lag7_log"]  - meta["lag7_mean"])   / meta["lag7_std"]
    roll7_z = (feat["roll7_log"] - meta["roll7_mean"])  / meta["roll7_std"]
    roll14_z= (feat["roll14_log"]- meta["roll14_mean"]) / meta["roll14_std"]
    sdev7_z = (feat["sdiff7_log"]- meta["sdev_mean"])   / meta["sdev_std"]

    cont = np.stack([
        feat["dow_sin"].astype(np.float32),
        feat["dow_cos"].astype(np.float32),
        feat["mon_sin"].astype(np.float32),
        feat["mon_cos"].astype(np.float32),
        feat["doy_sin"].astype(np.float32),
        feat["doy_cos"].astype(np.float32),
        lag1_z.astype(np.float32), lag7_z.astype(np.float32),
        roll7_z.astype(np.float32), roll14_z.astype(np.float32), sdev7_z.astype(np.float32),
    ], axis=1)

    y_log = feat["y_log"].to_numpy().astype(np.float32)
    ts    = pd.to_datetime(feat["ts"]).to_numpy()
    roll7 = feat["roll7_log"].to_numpy().astype(np.float32)
    return cat, cont, y_log, ts, roll7

def _build_targets_within_split(y_log: np.ndarray, roll7_log_t: np.ndarray, horizons=HORIZONS):
    """Compute targets inside a split so we never look across split boundaries."""
    N = len(y_log); H = len(horizons)
    Y = np.full((N, H), np.nan, dtype=np.float32)
    for j, h in enumerate(horizons):
        y_future = np.r_[y_log[h:], np.full(h, np.nan, dtype=np.float32)]
        Y[:, j] = y_future - roll7_log_t
    valid = np.all(~np.isnan(Y), axis=1)
    return Y[valid], valid

def prepare_data(batch_size=256, train_frac=0.70, val_frac=0.15):
    raw = load_csv(DATA_CSV)
    daily = make_daily(raw)

    # Chronological split by row index (already sorted by date)
    n = len(daily)
    tr_n = int(train_frac*n)
    va_n = int((train_frac+val_frac)*n)

    daily_tr = daily.iloc[:tr_n].copy()
    daily_va = daily.iloc[tr_n:va_n].copy()
    daily_te = daily.iloc[va_n:].copy()

    # Fit standardization on TRAIN only
    meta = standardize_hist(daily_tr)

    # Encode each split with TRAIN stats
    cat_tr, cont_tr, ylog_tr, ts_tr, roll7_tr = _encode_with_meta(daily_tr, meta)
    cat_va, cont_va, ylog_va, ts_va, roll7_va = _encode_with_meta(daily_va, meta)
    cat_te, cont_te, ylog_te, ts_te, roll7_te = _encode_with_meta(daily_te, meta)

    # Build residual targets within each split (prevents label leakage across boundaries)
    T_tr, m_tr = _build_targets_within_split(ylog_tr, roll7_tr, HORIZONS)
    T_va, m_va = _build_targets_within_split(ylog_va, roll7_va, HORIZONS)
    T_te, m_te = _build_targets_within_split(ylog_te, roll7_te, HORIZONS)

    # Trim arrays to valid rows per split
    def trim(cat, cont, roll7, ylog, ts, mask):
        return (cat[mask], cont[mask], roll7[mask], ylog[mask], ts[mask])

    Xc_tr, Xn_tr, ro_tr, yl_tr, ts_tr = trim(cat_tr, cont_tr, roll7_tr, ylog_tr, ts_tr, m_tr)
    Xc_va, Xn_va, ro_va, yl_va, ts_va = trim(cat_va, cont_va, roll7_va, ylog_va, ts_va, m_va)
    Xc_te, Xn_te, ro_te, yl_te, ts_te = trim(cat_te, cont_te, roll7_te, ylog_te, ts_te, m_te)

    # Datasets / loaders
    def mk(cat, cont, roll7, ylog, ts, T):
        return TicketHorizonDataset(cat, cont, roll7, ylog, ts, T)

    train_ds = mk(Xc_tr, Xn_tr, ro_tr, yl_tr, ts_tr, T_tr)
    val_ds   = mk(Xc_va, Xn_va, ro_va, yl_va, ts_va, T_va)
    test_ds  = mk(Xc_te, Xn_te, ro_te, yl_te, ts_te, T_te)

    loaders = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    )

    cardinals = dict(dow=7, mon=12, wk=2, hol=2)
    extra = dict(ts_train=ts_tr, ts_val=ts_va, ts_test=ts_te, daily=daily)
    return loaders, meta, cardinals, extra


def epoch_val_metrics(model, loader, device=DEVICE):
    model.eval()
    per_h = {h: {"y": [], "yp": []} for h in HORIZONS}
    with torch.no_grad():
        for cat, cont, roll7, _ylog, targets in loader:
            cat, cont = cat.to(device), cont.to(device)
            res_hat = model(cat, cont).cpu().numpy()  # [B,H]
            roll7   = roll7.numpy()                   # [B]
            tgt     = targets.numpy()                 # [B,H]
            for j, h in enumerate(HORIZONS):
                y_pred = np.expm1(roll7 + res_hat[:, j])
                y_true = np.expm1(roll7 + tgt[:, j])
                per_h[h]["y"].append(y_true); per_h[h]["yp"].append(y_pred)

    metrics = {}
    for h in HORIZONS:
        yt = np.concatenate(per_h[h]["y"]) if per_h[h]["y"] else np.array([])
        yp = np.concatenate(per_h[h]["yp"]) if per_h[h]["yp"] else np.array([])
        if yt.size == 0:
            metrics[h] = dict(mae=np.nan, rmse=np.nan, smape=np.nan, wape=np.nan, r2=None)
            continue
        metrics[h] = dict(
            mae=float(mean_absolute_error(yt, yp)),
            rmse=float(math.sqrt(mean_squared_error(yt, yp))),
            smape=float(smape(yt, yp)),
            wape=float(wape(yt, yp)),
            r2=float(r2_score(yt, yp)) if yt.size > 1 else None
        )
    return metrics


def _dicts_from_outputs(outputs_by_h):
    """Convert collect_predictions(...) result into y_true_dict/y_pred_dict keyed by horizon."""
    y_true_dict = {h: dfh["y_true"].to_numpy() for h, dfh in outputs_by_h.items()}
    y_pred_dict = {h: dfh["y_pred"].to_numpy() for h, dfh in outputs_by_h.items()}
    return y_true_dict, y_pred_dict


def compute_overall_metrics(y_true_dict, y_pred_dict, horizon_metrics=None):

    # pool all horizons, then compute once
    y_true_all = np.concatenate(list(y_true_dict.values()))
    y_pred_all = np.concatenate(list(y_pred_dict.values()))

    mae  = mean_absolute_error(y_true_all, y_pred_all)
    rmse = mean_squared_error(y_true_all, y_pred_all)
    smape_val = 100 * np.mean(
        2 * np.abs(y_pred_all - y_true_all) /
        (np.abs(y_true_all) + np.abs(y_pred_all) + 1e-8)
    )
    wape_val = 100 * np.sum(np.abs(y_pred_all - y_true_all)) / np.sum(np.abs(y_true_all))
    r2   = r2_score(y_true_all, y_pred_all)

    results = {"micro": dict(mae=mae, rmse=rmse, smape=smape_val, wape=wape_val, r2=r2)}

    #  Macro-average (if horizon-level metrics already provided)
    if horizon_metrics is not None:
        macro = {}
        for k in ["mae", "rmse", "smape", "wape", "r2"]:
            macro[k] = float(np.mean([horizon_metrics[h][k] for h in horizon_metrics]))
        results["macro"] = macro

    return results



# -----------------------------
# Plot helpers
# -----------------------------
def plot_horizon(dfh, h, out_path, title):
    dfh = dfh.sort_values("date")
    plt.figure(figsize=(12,6))
    plt.plot(dfh["date"], dfh["y_true"], label="Actual")
    plt.plot(dfh["date"], dfh["y_pred"], label="Predicted", linestyle="--")
    plt.xlabel("Date"); plt.ylabel("Daily tickets"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

def plot_train(history, out_path):
    ep = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 6))

    ax1 = plt.gca()
    l1, = ax1.plot(
        ep, history["train_loss"], label="Train MSE on residuals", color="blue"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss (MSE)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l2, = ax2.plot(
        ep, history["val_wape"], label="Val WAPE (avg over h)", color="orange", linestyle="--"
    )
    ax2.set_ylabel("Validation WAPE (%)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # combine legends
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Training / Validation Curves")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Train / Evaluate
# -----------------------------
def train(epochs=100, lr=1e-3, patience=12, min_delta=0.0, clip_grad=1.0, batch_size=256):
    (train_loader, val_loader, test_loader), meta, cardinals, extra = prepare_data(batch_size)
    model = TabTransformerMultiH(cardinals, cont_dim=len(meta["cont_names"]), n_horizons=len(HORIZONS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=3e-5, verbose=True)
    criterion = nn.MSELoss()

    run_dir = make_run_dir()
    history = {"train_loss": [], "val_wape": []}
    best = None; best_score = float("inf"); no_improve = 0

    for ep in range(1, epochs+1):
        model.train(); running=0.0; nobs=0
        for cat, cont, roll7, ylog, targets in train_loader:
            cat, cont, targets = cat.to(DEVICE), cont.to(DEVICE), targets.to(DEVICE)
            opt.zero_grad()
            out = model(cat, cont)           
            loss = criterion(out, targets)
            loss.backward()
            if clip_grad is not None: nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            running += loss.item()*len(cat); nobs += len(cat)
        train_loss = running/max(nobs,1)

        # validation summary: average WAPE across horizons
        vmet = epoch_val_metrics(model, val_loader)
        wapes = [vmet[h]["wape"] for h in HORIZONS]
        val_wape = float(np.mean(wapes))

        history["train_loss"].append(train_loss); history["val_wape"].append(val_wape)
        print(f"Epoch {ep:03d} | train MSE={train_loss:.4f} | val WAPE(avg)={val_wape:.2f}%  "
              + " ".join([f"h{h}:WAPE={vmet[h]['wape']:.2f}%" for h in HORIZONS]))

        sched.step(val_wape)
        if val_wape + min_delta < best_score:
            best_score = val_wape; best = model.state_dict(); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep}. Best avg WAPE={best_score:.2f}%.")
                break

    if best is not None: model.load_state_dict(best)

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)

    # Final eval (val + test) with CSVs & plots for each horizon
    for split_name, loader, ts in [("val", val_loader, extra["ts_val"]), ("test", test_loader, extra["ts_test"])]:
        out = collect_predictions(model, loader, ts)
        # Save and plot
        for h, dfh in out.items():
            path_csv = os.path.join(run_dir, f"{split_name}_h{h}.csv")
            dfh.to_csv(path_csv, index=False)
            plot_horizon(dfh, h, os.path.join(run_dir, f"{split_name}_h{h}.png"),
                         title=f"{split_name.title()} Window: Actual vs Predicted (h=+{h}d)")

    #  Save metrics for val and test 
    val_per_h  = epoch_val_metrics(model, val_loader)
    test_per_h = epoch_val_metrics(model, test_loader)

    # Build overall metrics from predictions
    val_outputs  = collect_predictions(model, val_loader,  extra["ts_val"])
    test_outputs = collect_predictions(model, test_loader, extra["ts_test"])

    val_y_true,  val_y_pred  = _dicts_from_outputs(val_outputs)
    test_y_true, test_y_pred = _dicts_from_outputs(test_outputs)

    val_overall  = compute_overall_metrics(val_y_true,  val_y_pred,  horizon_metrics=val_per_h)
    test_overall = compute_overall_metrics(test_y_true, test_y_pred, horizon_metrics=test_per_h)

    metrics_payload = {
        "best_val_avg_wape": best_score,   
        "val": {
            "per_horizon": val_per_h,    
            "overall": val_overall 
        },
        "test": {
            "per_horizon": test_per_h,
            "overall": test_overall
        }
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save training curve
    plot_train(history, os.path.join(run_dir, "train_val_curves.png"))

    print(f"\nArtifacts saved to: {run_dir}\n(also linked at ./artifacts/latest)")
    return run_dir

def collect_predictions(model, loader, ts_at_t):
    """Return dict[h] -> DataFrame(date=t+h, y_true, y_pred). Assumes loader shuffle=False."""
    model.eval()
    buff = {h: [] for h in HORIZONS}
    with torch.no_grad():
        idx = 0
        for batch in loader:
            cat, cont, roll7, _ylog, targets = batch
            B = len(cat)
            cat, cont = cat.to(DEVICE), cont.to(DEVICE)
            res_hat = model(cat, cont).cpu().numpy() 
            roll7   = roll7.numpy()  
            targets = targets.numpy()    
            # dates at t (ordered)
            dates_t = pd.to_datetime(ts_at_t[idx:idx+B])
            for j, h in enumerate(HORIZONS):
                y_pred = np.expm1(roll7 + res_hat[:, j])
                y_true = np.expm1(roll7 + targets[:, j])
                dates_h = (dates_t + pd.to_timedelta(h, unit="D")).to_numpy()
                buff[h].append(pd.DataFrame({"date": dates_h, "y_true": y_true, "y_pred": y_pred}))
            idx += B
    return {h: pd.concat(frames).sort_values("date") for h, frames in buff.items()}

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Direct multi-horizon (+1,+3,+7d) TabTransformer")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    args = ap.parse_args()

    train(epochs=args.epochs, lr=args.lr, patience=args.patience,
          min_delta=args.min_delta, clip_grad=args.clip_grad,
          batch_size=args.batch_size)

if __name__ == "__main__":
    main()
