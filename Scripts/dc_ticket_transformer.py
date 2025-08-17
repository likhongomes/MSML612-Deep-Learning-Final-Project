#!/usr/bin/env python3
import argparse
import math
import os
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

import glob

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DATA_PATH = "cleaned_parking_violations_v2.csv"
MODEL_PATH = "dc_ticket_transformer.pt"
ENCODER_PATH = "dc_ticket_feature_meta.npz"

# -----------------------------
# Data utilities
# -----------------------------

def load_parking_violation_data(data_folder):
        files = os.listdir(data_folder)
        print(files)
        all_csvs = glob.glob(os.path.join(data_folder, "*.csv"))
        dfs = []
        for c in all_csvs:
            try:
                df = pd.read_csv(c)
                dfs.append(df)
                print(f"successfully loaded {c}")
            except Exception as e:
                print(f"error opening {c}: {e}")
        return pd.concat(dfs, ignore_index=True)

def load_clean_df(path=DATA_PATH) -> pd.DataFrame:
    

    # df = pd.read_csv(path, low_memory=False)

    df = load_parking_violation_data("../CleanData")
    # Parse datetime
    if "issue_datetime" in df.columns:
        df["issue_datetime"] = pd.to_datetime(df["issue_datetime"], errors="coerce", utc=True)
    else:
        # If different column name, try to infer
        dt_col = None
        for c in df.columns:
            if "date" in c.lower() and "time" in c.lower():
                dt_col = c
                break
        if dt_col is None:
            raise ValueError("Could not find issue_datetime column")
        df["issue_datetime"] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)

    # Basic column sanity
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Expected latitude/longitude columns")

    # Drop missing coords or datetimes
    df = df.dropna(subset=["latitude", "longitude", "issue_datetime"]).copy()

    # Ensure numeric
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Keep only plausible DC bounds (rough bounding box)
    # DC approx: lat 38.78..39.03, lon -77.12..-76.90
    df = df[(df["latitude"].between(38.78, 39.03)) & (df["longitude"].between(-77.12, -76.90))].copy()

    # Derive time features if not present
    df["hour"] = df["issue_datetime"].dt.hour
    df["day_of_week"] = df["issue_datetime"].dt.dayofweek  # Mon=0
    df["month"] = df["issue_datetime"].dt.month
    df["date"] = df["issue_datetime"].dt.date

    # Create coarse grid (about ~110m) by rounding ~3 decimals
    df["lat_grid"] = df["latitude"].round(3)
    df["lon_grid"] = df["longitude"].round(3)

    # Create hour bucket (0-23)
    df["hour_bucket"] = df["hour"]

    # Use only the columns we need downstream
    return df[["issue_datetime","latitude","longitude","lat_grid","lon_grid","hour_bucket","day_of_week","month","date"]].copy()


def build_positive_negative_samples(df: pd.DataFrame, max_neg_per_pos: float = 1.0) -> pd.DataFrame:
    """
    Build binary samples using grid x hour-bucket occurrences as positives.
    Negatives are sampled from grid-hour cells where no ticket occurred that day.
    """
    # Positive keys: (date, lat_grid, lon_grid, hour_bucket)
    df["key"] = list(zip(df["date"], df["lat_grid"], df["lon_grid"], df["hour_bucket"]))
    pos = df.groupby(["date","lat_grid","lon_grid","hour_bucket"], as_index=False).size()
    pos["label"] = 1

    # Universe of candidate keys for negatives: for each date, use the observed set of grid cells that day to avoid sampling unvisited areas
    # Build per-date grids
    daily_grids = df.groupby("date")[["lat_grid","lon_grid"]].apply(lambda g: g.drop_duplicates().reset_index(drop=True)).reset_index()
    # Expand with all 24 hours
    rows = []
    for date, grid_df in df.groupby("date")[["lat_grid", "lon_grid"]]:
        grid_df = grid_df.drop_duplicates().reset_index(drop=True)
        for _, gg in grid_df.iterrows():
            for h in range(24):
                rows.append((
                    date,
                    float(gg["lat_grid"]),
                    float(gg["lon_grid"]),
                    h
                ))
    full = pd.DataFrame(rows, columns=["date", "lat_grid", "lon_grid", "hour_bucket"])

    # Mark positives
    full = full.merge(pos[["date","lat_grid","lon_grid","hour_bucket","label"]], how="left",
                      on=["date","lat_grid","lon_grid","hour_bucket"])
    full["label"] = full["label"].fillna(0).astype(int)

    # Downsample negatives to achieve ~max_neg_per_pos ratio
    num_pos = (full["label"] == 1).sum()
    neg = full[full["label"] == 0]
    if num_pos == 0:
        raise ValueError("No positive samples found. Check the dataset time window.")
    target_neg = int(min(len(neg), num_pos * max_neg_per_pos))
    neg_sampled = neg.sample(n=target_neg, random_state=RANDOM_SEED) if target_neg < len(neg) else neg

    pos_only = full[full["label"] == 1]
    data = pd.concat([pos_only, neg_sampled], axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # Attach representative lat/lon for each (date, grid, hour) by taking medians from df
    rep = df.groupby(["date","lat_grid","lon_grid","hour_bucket"]).agg(
        latitude=("latitude","median"),
        longitude=("longitude","median"),
        day_of_week=("day_of_week","median"),
        month=("month","median"),
    ).reset_index()

    data = data.merge(rep, on=["date","lat_grid","lon_grid","hour_bucket"], how="left")
    # For negatives (no rep), fill with exact grid center
    data["latitude"] = data["latitude"].fillna(data["lat_grid"])
    data["longitude"] = data["longitude"].fillna(data["lon_grid"])
    # day_of_week/month for negatives: derive from 'date'
    missing_mask = data["day_of_week"].isna()
    if missing_mask.any():
        dts = pd.to_datetime(data.loc[missing_mask, "date"])
        data.loc[missing_mask, "day_of_week"] = dts.dt.dayofweek.values
        data.loc[missing_mask, "month"] = dts.dt.month.values

    data["day_of_week"] = data["day_of_week"].astype(int)
    data["month"] = data["month"].astype(int)

    return data


# -----------------------------
# Feature encoding
# -----------------------------

def fit_feature_meta(df: pd.DataFrame) -> dict:
    meta = {}
    # Continuous ranges (for simple normalization)
    meta["lat_min"], meta["lat_max"] = df["latitude"].min(), df["latitude"].max()
    meta["lon_min"], meta["lon_max"] = df["longitude"].min(), df["longitude"].max()

    # Categories
    meta["hour_cardinality"] = 24
    meta["dow_cardinality"] = 7
    meta["month_cardinality"] = 12

    return meta


def save_feature_meta(meta: dict, path: str = ENCODER_PATH):
    np.savez(path, **meta)


def load_feature_meta(path: str = ENCODER_PATH) -> dict:
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k].item() if npz[k].shape == () else npz[k] for k in npz.files}


def encode_features(df: pd.DataFrame, meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      cat_tokens: [N, 4] with (hour, dow, month, grid_id)
      cont_feats: [N, C] with continuous features (lat_norm, lon_norm, sin/cos hour, sin/cos dow, sin/cos month)
      labels: [N]
    """
    # Build a grid id for categorical tokenization
    # Map (lat_grid, lon_grid) pairs to ids
    grid_pairs = df[["lat_grid","lon_grid"]].drop_duplicates().reset_index(drop=True)
    grid_pairs["grid_id"] = np.arange(len(grid_pairs))
    df = df.merge(grid_pairs, on=["lat_grid","lon_grid"], how="left")

    # Normalize lat/lon
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    lat_norm = (lat - meta["lat_min"]) / max(1e-6, (meta["lat_max"] - meta["lat_min"])) * 2 - 1
    lon_norm = (lon - meta["lon_min"]) / max(1e-6, (meta["lon_max"] - meta["lon_min"])) * 2 - 1

    # Cyclical encodings
    hour = df["hour_bucket"].to_numpy().astype(int)
    dow = df["day_of_week"].to_numpy().astype(int)
    month = df["month"].to_numpy().astype(int)

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    month_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12.0)

    cont = np.stack([lat_norm, lon_norm, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos], axis=1)

    # Categorical tokens: hour, dow, month, grid_id
    cat = np.stack([hour, dow, month - 1, df["grid_id"].to_numpy().astype(int)], axis=1)

    labels = df["label"].to_numpy().astype(np.float32)

    # Save grid_pairs to meta for inference mapping
    meta["grid_lat_lon"] = grid_pairs[["lat_grid","lon_grid","grid_id"]].to_dict(orient="list")

    return cat, cont.astype(np.float32), labels


# -----------------------------
# Dataset
# -----------------------------

class TicketDataset(Dataset):
    def __init__(self, cat_tokens, cont_feats, labels):
        self.cat = torch.tensor(cat_tokens, dtype=torch.long)
        self.cont = torch.tensor(cont_feats, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx], self.y[idx]


# -----------------------------
# Model (TabTransformer-style)
# -----------------------------

class TabTransformer(nn.Module):
    def __init__(self,
                 hour_card=24,
                 dow_card=7,
                 month_card=12,
                 grid_card=10000,
                 d_token=32,
                 n_heads=4,
                 n_layers=2,
                 cont_dim=8,
                 mlp_hidden=64,
                 dropout=0.1):
        super().__init__()
        # Embeddings for categorical tokens
        self.hour_emb = nn.Embedding(hour_card, d_token)
        self.dow_emb = nn.Embedding(dow_card, d_token)
        self.month_emb = nn.Embedding(month_card, d_token)
        self.grid_emb = nn.Embedding(grid_card, d_token)

        # Transformer encoder over the token sequence [hour, dow, month, grid]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Continuous MLP projection
        self.cont_proj = nn.Sequential(
            nn.Linear(cont_dim, d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final head: pool tokens + cont proj, then MLP -> sigmoid
        self.head = nn.Sequential(
            nn.Linear(d_token*5, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, cat_tokens, cont_feats):
        # cat_tokens: [B, 4]
        h = self.hour_emb(cat_tokens[:,0])
        d = self.dow_emb(cat_tokens[:,1])
        m = self.month_emb(cat_tokens[:,2])
        g = self.grid_emb(cat_tokens[:,3])

        tokens = torch.stack([h,d,m,g], dim=1)  # [B, 4, d_token]
        z = self.encoder(tokens)                # [B, 4, d_token]
        z_pool = z.mean(dim=1)                  # [B, d_token]

        c = self.cont_proj(cont_feats)          # [B, d_token]

        x = torch.cat([z_pool, h, d, m, c], dim=1)  # concat pooled + individual + cont

        logits = self.head(x).squeeze(1)
        return logits


# -----------------------------
# Training / Eval
# -----------------------------

def train_model(train_loader, val_loader, meta, grid_card, epochs=5, lr=1e-3, device="cpu"):
    model = TabTransformer(hour_card=24, dow_card=7, month_card=12, grid_card=grid_card,
                           d_token=48, n_heads=4, n_layers=2, cont_dim=8, mlp_hidden=96, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for cat, cont, y in train_loader:
            cat, cont, y = cat.to(device), cont.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(cat, cont)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        avg_loss = total_loss / len(train_loader.dataset)

        # Eval
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for cat, cont, y in val_loader:
                cat, cont = cat.to(device), cont.to(device)
                logits = model(cat, cont)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_p.append(probs)
                all_y.append(y.numpy())
        y_true = np.concatenate(all_y)
        y_prob = np.concatenate(all_p)
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} val_auc={auc:.4f} val_ap={ap:.4f}")

        mae  = mean_absolute_error(y_true, y_prob)
        rmse = np.sqrt(mean_squared_error(y_true, y_prob))
        mape_val = mape(y_true, y_prob)

        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape_val:.2f}%")

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    torch.save(model.state_dict(), MODEL_PATH)
    save_feature_meta(meta)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved encoder/meta to {ENCODER_PATH}")
    return model, best_auc


def prepare_dataloaders(max_neg_per_pos=1.0, batch_size=1024, test_size=0.2, val_size=0.1):
    df = load_clean_df(DATA_PATH)
    data = build_positive_negative_samples(df, max_neg_per_pos=max_neg_per_pos)

    # Fit meta and encode
    meta = fit_feature_meta(data)
    cat, cont, labels = encode_features(data, meta)

    # Determine grid_cardinality
    grid_card = int(cat[:,3].max()) + 1

    X_cat_train, X_cat_tmp, X_cont_train, X_cont_tmp, y_train, y_tmp = train_test_split(
        cat, cont, labels, test_size=test_size, random_state=RANDOM_SEED, stratify=labels
    )
    # Split tmp into val/test
    val_ratio = val_size / (1 - test_size)
    X_cat_val, X_cat_test, X_cont_val, X_cont_test, y_val, y_test = train_test_split(
        X_cat_tmp, X_cont_tmp, y_tmp, test_size=1 - val_ratio, random_state=RANDOM_SEED, stratify=y_tmp
    )

    train_ds = TicketDataset(X_cat_train, X_cont_train, y_train)
    val_ds = TicketDataset(X_cat_val, X_cont_val, y_val)
    test_ds = TicketDataset(X_cat_test, X_cont_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, meta, grid_card


def eval_model(model, test_loader, device="cpu"):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for cat, cont, y in test_loader:
            cat, cont = cat.to(device), cont.to(device)
            logits = model(cat, cont)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_p.append(probs)
            all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_p)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    print("TEST AUC:", round(auc,4), "TEST AP:", round(ap,4))
    print(classification_report((y_prob>0.5).astype(int), y_true.astype(int)))

    mae  = mean_absolute_error(y_true, y_prob)
    rmse = np.sqrt(mean_squared_error(y_true, y_prob))
    mape_val = mape(y_true, y_prob)

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape_val:.2f}%")


# -----------------------------
# Inference utility
# -----------------------------

def load_model_for_inference():
    meta = load_feature_meta(ENCODER_PATH)
    # Reconstruct grid mapping
    grid_lists = meta["grid_lat_lon"]
    # Build a dataframe for easy nearest grid mapping
    grid_df = pd.DataFrame({
        "lat_grid": grid_lists["lat_grid"],
        "lon_grid": grid_lists["lon_grid"],
        "grid_id": grid_lists["grid_id"]
    })
    # Model
    grid_card = int(max(grid_df["grid_id"])) + 1
    model = TabTransformer(hour_card=24, dow_card=7, month_card=12, grid_card=grid_card,
                           d_token=48, n_heads=4, n_layers=2, cont_dim=8, mlp_hidden=96, dropout=0.1)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta, grid_df


def _nearest_grid(lat, lon, grid_df: pd.DataFrame):
    # Round to 3 decimals to find the matching grid; if missing, snap to nearest
    lat_g = round(lat, 3)
    lon_g = round(lon, 3)
    match = grid_df[(grid_df["lat_grid"]==lat_g) & (grid_df["lon_grid"]==lon_g)]
    if len(match) == 0:
        # nearest by simple L2 in grid space
        diff = (grid_df["lat_grid"] - lat_g)**2 + (grid_df["lon_grid"] - lon_g)**2
        idx = diff.idxmin()
        return int(grid_df.loc[idx, "grid_id"]), float(grid_df.loc[idx, "lat_grid"]), float(grid_df.loc[idx, "lon_grid"])
    else:
        return int(match.iloc[0]["grid_id"]), lat_g, lon_g


def predict_probability(lat: float, lon: float, when_iso: str) -> float:
    """
    lat, lon in decimal degrees
    when_iso: ISO-8601 datetime (assumed local; no TZ conversion applied)
    """
    model, meta, grid_df = load_model_for_inference()
    # Parse time
    when = pd.to_datetime(when_iso, errors="coerce")
    if pd.isna(when):
        raise ValueError("Invalid datetime format")

    hour = int(when.hour)
    dow = int(when.dayofweek)
    month = int(when.month)

    grid_id, lat_g, lon_g = _nearest_grid(lat, lon, grid_df)

    # Build feature tensors
    lat_norm = (lat - meta["lat_min"]) / max(1e-6, (meta["lat_max"] - meta["lat_min"])) * 2 - 1
    lon_norm = (lon - meta["lon_min"]) / max(1e-6, (meta["lon_max"] - meta["lon_min"])) * 2 - 1

    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    dow_sin = math.sin(2 * math.pi * dow / 7.0)
    dow_cos = math.cos(2 * math.pi * dow / 7.0)
    month_sin = math.sin(2 * math.pi * (month - 1) / 12.0)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12.0)

    cont = torch.tensor([[lat_norm, lon_norm, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos]], dtype=torch.float32)
    cat = torch.tensor([[hour, dow, month-1, grid_id]], dtype=torch.long)

    with torch.no_grad():
        logit = model(cat, cont)
        prob = torch.sigmoid(logit).item()
    return float(prob)

def predict_by_parts(lat: float, lon: float, hour: int, dow: int, month: int | None = None) -> float:
    """
    Predict using explicit parts instead of a datetime string.
      - hour: 0..23
      - dow: 0..6 (Mon=0)
      - month: 1..12 (optional). If None, uses the most common month from training.
    """
    model, meta, grid_df = load_model_for_inference()

    if month is None:
        month = int(meta.get("month_mode", 6))

    # Validate ranges
    if not (0 <= hour <= 23):
        raise ValueError("hour must be in 0..23")
    if not (0 <= dow <= 6):
        raise ValueError("dow must be in 0..6 (Mon=0)")
    if not (1 <= month <= 12):
        raise ValueError("month must be in 1..12")

    # Nearest grid id
    grid_id, _, _ = _nearest_grid(lat, lon, grid_df)

    # Normalize continuous features
    lat_norm = (lat - meta["lat_min"]) / max(1e-6, (meta["lat_max"] - meta["lat_min"])) * 2 - 1
    lon_norm = (lon - meta["lon_min"]) / max(1e-6, (meta["lon_max"] - meta["lon_min"])) * 2 - 1

    import math
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    dow_sin = math.sin(2 * math.pi * dow / 7.0)
    dow_cos = math.cos(2 * math.pi * dow / 7.0)
    month_sin = math.sin(2 * math.pi * (month - 1) / 12.0)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12.0)

    cont = torch.tensor(
        [[lat_norm, lon_norm, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos]],
        dtype=torch.float32
    )
    cat = torch.tensor(
        [[hour, dow, month-1, grid_id]],
        dtype=torch.long
    )

    with torch.no_grad():
        logit = model(cat, cont)
        prob = torch.sigmoid(logit).item()
    return float(prob)

def mape(y_true, y_pred, eps=1e-6):
    # avoid exploding errors when y_true has zeros
    # denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    # return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="DC Parking Ticket TabTransformer")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--max-neg-per-pos", type=float, default=1.0, help="Negative sampling ratio")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=1024)
    p_train.add_argument("--lr", type=float, default=1e-3)

    p_eval = subparsers.add_parser("eval", help="Evaluate the saved model")

    p_pred = subparsers.add_parser("predict", help="Predict probability for a lat/lon/time")
    p_pred.add_argument("--lat", type=float, required=True)
    p_pred.add_argument("--lon", type=float, required=True)
    p_pred.add_argument("--when", type=str, required=False, help="ISO datetime, e.g., '2025-08-09 14:30'")

    p_pred.add_argument("--hour", type=str, required=False, help="ISO datetime, e.g., '23'")
    p_pred.add_argument("--dow", type=str, required=False, help="ISO datetime, e.g., 'Sunday'")
    p_pred.add_argument("--month", type=str, required=False, help="ISO datetime, e.g., '6'")

    args = parser.parse_args()

    if args.cmd == "train":
        train_loader, val_loader, test_loader, meta, grid_card = prepare_dataloaders(
            max_neg_per_pos=args.max_neg_per_pos, batch_size=args.batch_size
        )
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model, best_auc = train_model(train_loader, val_loader, meta, grid_card, epochs=args.epochs, lr=args.lr, device=device)
        eval_model(model, test_loader, device=device)
    elif args.cmd == "eval":
        # Rebuild loaders just for test split metrics
        _, _, test_loader, meta, grid_card = prepare_dataloaders()
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = TabTransformer(hour_card=24, dow_card=7, month_card=12, grid_card=grid_card,
                               d_token=48, n_heads=4, n_layers=2, cont_dim=8, mlp_hidden=96, dropout=0.1).to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        eval_model(model, test_loader, device=device)
    elif args.cmd == "predict":
        # 
        if args.when is not None:
            prob = predict_probability(args.lat, args.lon, args.when)
        else:
            prob = predict_by_parts(args.lat, args.lon, int(args.hour), int(args.dow), month=int(args.month) if args.month else None)

        print(f"Predicted probability: {prob*100:.2f}%")

if __name__ == "__main__":
    main()
