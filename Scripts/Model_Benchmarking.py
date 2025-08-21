# pip install -q pandas numpy scikit-learn torch tab-transformer-pytorch
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tab_transformer_pytorch import TabTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "../CleanData/cleaned_parking_violations_v2.csv"          # your raw ticket events file
TIME_COL = "issue_datetime"       # timestamp column in your dataset
LAT_COL, LON_COL = "latitude", "longitude"
LOCATION_COL = "location"         # can be a street/block id; if None, citywide aggregation
FREQ = "D"                        # "D" for daily. Use "H" for hourly.
HORIZONS = [1, 3, 7]              # t+1, t+3, t+7
MAX_LAG = 28                      # how many past days to use as features
ROLL_WINDOWS = [7, 14, 28]        # rolling windows for mean features
BATCH_SIZE = 512
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# -------------------------------
# 1) LOAD + AGGREGATE TICKETS → COUNTS
# -------------------------------
df = pd.read_csv(CSV_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])

# Choose citywide or per-location modeling:
# - Citywide: counts per day across all locations (simplest, matches many SARIMAX setups)
# - Per location: set USE_LOCATION = True and pick a specific location_id to model.
USE_LOCATION = False
location_id = None

if USE_LOCATION and LOCATION_COL in df.columns:
    # pick the top location by volume if not specified
    if location_id is None:
        location_id = df[LOCATION_COL].value_counts().index[0]
    df = df[df[LOCATION_COL] == location_id].copy()

# resample to daily (or hourly) counts
counts = (
    df.set_index(TIME_COL)
      .sort_index()
      .assign(count=1)
      .resample(FREQ)["count"].sum()
      .to_frame("y")
)

# fill missing dates with 0 tickets (common for count series)
counts["y"] = counts["y"].fillna(0).astype(float)

# -------------------------------
# 2) FEATURE ENGINEERING
# -------------------------------
X = counts.copy()
X["year"] = X.index.year
if FREQ == "D":
    X["month"] = X.index.month
    X["dayofweek"] = X.index.dayofweek
    X["is_weekend"] = (X["dayofweek"] >= 5).astype(int)
    # seasonal/cyclical encodings
    X["dow_sin"] = np.sin(2*np.pi*X["dayofweek"]/7)
    X["dow_cos"] = np.cos(2*np.pi*X["dayofweek"]/7)
    X["doy"] = X.index.dayofyear
    X["doy_sin"] = np.sin(2*np.pi*X["doy"]/365.25)
    X["doy_cos"] = np.cos(2*np.pi*X["doy"]/365.25)
elif FREQ == "H":
    X["hour"] = X.index.hour
    X["dayofweek"] = X.index.dayofweek
    X["is_weekend"] = (X["dayofweek"] >= 5).astype(int)
    X["hour_sin"] = np.sin(2*np.pi*X["hour"]/24)
    X["hour_cos"] = np.cos(2*np.pi*X["hour"]/24)
    X["dow_sin"] = np.sin(2*np.pi*X["dayofweek"]/7)
    X["dow_cos"] = np.cos(2*np.pi*X["dayofweek"]/7)

# lags
for lag in range(1, MAX_LAG + 1):
    X[f"lag_{lag}"] = X["y"].shift(lag)

# rolling stats on original y (shifted by 1 to avoid leakage)
for w in ROLL_WINDOWS:
    X[f"roll_mean_{w}"] = X["y"].shift(1).rolling(w).mean()
    X[f"roll_std_{w}"]  = X["y"].shift(1).rolling(w).std()

# drop initial NaNs from lags/rolls
X = X.dropna().copy()

# -------------------------------
# 3) BUILD SUPERVISED SETS FOR EACH HORIZON
#    y_{t+h} as target, features are at time t
# -------------------------------
def make_xy_for_horizon(frame: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    y_target = frame["y"].shift(-horizon)  # target at t+h
    df_feat = frame.drop(columns=["y"]).iloc[:-horizon, :].copy()
    y_out   = y_target.iloc[:-horizon].copy()
    return df_feat, y_out

# Choose categorical and continuous features for TabTransformer
if FREQ == "D":
    cat_cols = ["dayofweek", "month", "is_weekend"]
    cont_cols = [c for c in X.columns if c not in ["y"] + cat_cols]
elif FREQ == "H":
    cat_cols = ["dayofweek", "is_weekend", "hour"]
    cont_cols = [c for c in X.columns if c not in ["y"] + cat_cols]

# -------------------------------
# 4) TIME-BASED TRAIN/VAL/TEST SPLIT
# -------------------------------
def time_split_index(n: int, train_frac=0.7, val_frac=0.15):
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]

def df_to_tensors(df_feat: pd.DataFrame, y: pd.Series,
                  cat_cols: List[str], cont_cols: List[str],
                  fitted_cats: Optional[Dict[str, pd.Index]] = None,
                  cont_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    # prepare categoricals
    cats_info = {}
    Xc = pd.DataFrame(index=df_feat.index)
    for col in cat_cols:
        if fitted_cats is None:
            cats = df_feat[col].astype("category").cat.categories
            cats_info[col] = cats
            Xc[col] = df_feat[col].astype("category").cat.codes
        else:
            cats = fitted_cats[col]
            Xc[col] = pd.Categorical(df_feat[col], categories=cats).codes
            cats_info[col] = cats

    # continuous normalization
    Xn_raw = df_feat[cont_cols].astype(float).values
    if cont_stats is None:
        means = np.nanmean(Xn_raw, axis=0)
        stds  = np.nanstd(Xn_raw, axis=0)
        stds[stds == 0] = 1.0
    else:
        means, stds = cont_stats
    Xn = (Xn_raw - means) / stds

    # replace any remaining NaNs
    Xn = np.nan_to_num(Xn)

    # tensors
    x_cat  = torch.tensor(Xc.values, dtype=torch.long)
    x_cont = torch.tensor(Xn, dtype=torch.float32)
    y_t    = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    return x_cat, x_cont, y_t, cats_info, (means, stds)

def train_tabtransformer_regression(xc_tr, xn_tr, y_tr, xc_va, xn_va, y_va,
                                    category_sizes: Tuple[int, ...],
                                    epochs=EPOCHS, lr=LR) -> TabTransformer:
    model = TabTransformer(
        categories=category_sizes,
        num_continuous=xn_tr.shape[1],
        dim=32,
        depth=4,
        heads=4,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=nn.ReLU(),
        dim_out=1
    ).to(DEVICE)

    crit = nn.MSELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(xc_tr, xn_tr, y_tr)
    val_ds   = TensorDataset(xc_va, xn_va, y_va)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for xc, xn, yy in train_dl:
            xc, xn, yy = xc.to(DEVICE), xn.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad()
            pred = model(xc, xn)
            loss = crit(pred, yy)
            loss.backward()
            opt.step()
            total += loss.item() * yy.size(0)

        # quick val mse
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xc, xn, yy in val_dl:
                xc, xn, yy = xc.to(DEVICE), xn.to(DEVICE), yy.to(DEVICE)
                pred = model(xc, xn)
                val_losses.append(crit(pred, yy).item() * yy.size(0))
        train_mse = total / len(train_ds)
        val_mse = sum(val_losses) / len(val_ds)
        print(f"Epoch {epoch:02d} | train_rmse={np.sqrt(train_mse):.2f} | val_rmse={np.sqrt(val_mse):.2f}")

    return model

def mape(y_true, y_pred, eps=1e-6):
    # avoid exploding errors when y_true has zeros
    # denom = np.maximum(np.abs(y_true), eps)

    # return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    # return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

# -------------------------------
# 5) RUN FOR EACH HORIZON & COLLECT METRICS
# -------------------------------
rows = []
for h in HORIZONS:
    print("\n" + "="*60)
    print(f"Training TabTransformer for horizon t+{h} ({'day' if FREQ=='D' else 'hour'})")
    feat_df, y_target = make_xy_for_horizon(X, h)

    # split by time
    n = len(feat_df)
    tr_idx, va_idx, te_idx = time_split_index(n, train_frac=0.7, val_frac=0.15)

    feat_tr, y_tr = feat_df.iloc[tr_idx], y_target.iloc[tr_idx]
    feat_va, y_va = feat_df.iloc[va_idx], y_target.iloc[va_idx]
    feat_te, y_te = feat_df.iloc[te_idx], y_target.iloc[te_idx]

    # fit encoders on train; apply to val/test
    xcat_tr, xcon_tr, yt_tr, cats_info, cont_stats = df_to_tensors(feat_tr, y_tr, cat_cols, cont_cols)
    # sizes for TabTransformer
    category_sizes = tuple(int(len(cats_info[c])) for c in cat_cols)

    # val/test tensors with fitted encoders/stats
    def encode(df_, y_):
        return df_to_tensors(df_, y_, cat_cols, cont_cols, fitted_cats={c: cats_info[c] for c in cat_cols},
                             cont_stats=cont_stats)

    xcat_va, xcon_va, yt_va, _, _ = encode(feat_va, y_va)
    xcat_te, xcon_te, yt_te, _, _ = encode(feat_te, y_te)

    # train model
    model = train_tabtransformer_regression(xcat_tr, xcon_tr, yt_tr, xcat_va, xcon_va, yt_va,
                                            category_sizes=category_sizes, epochs=EPOCHS, lr=LR)

    # evaluate on test
    model.eval()
    with torch.no_grad():
        yhat = model(xcat_te.to(DEVICE), xcon_te.to(DEVICE)).cpu().numpy().ravel()
    ytrue = yt_te.numpy().ravel()

    mae  = mean_absolute_error(ytrue, yhat)
    rmse = np.sqrt(mean_squared_error(ytrue, yhat))
    mape_val = mape(ytrue, yhat)

    rows.append({
        "Horizon": f"t+{h}",
        "Model (Best)": "TabTransformer",
        "MAE": round(mae, 1),
        "RMSE": round(rmse, 1),
        "MAPE": f"{mape_val:.1f}%"
    })

# -------------------------------
# 6) PRINT RESULTS TABLE
# -------------------------------
result_df = pd.DataFrame(rows, columns=["Horizon", "Model (Best)", "MAE", "RMSE", "MAPE"])
print("\nTabTransformer Results")
print(result_df.to_string(index=False))

# Optionally, save to CSV so you can merge with your existing table
result_df.to_csv("tabtransformer_metrics.csv", index=False)
print("\nSaved: tabtransformer_metrics.csv")
