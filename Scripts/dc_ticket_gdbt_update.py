import os, glob, json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

#  CONFIG 
CSV_PATH = "../CleanData/prepared_data.csv"   # folder containing CSVs
RANDOM_STATE = 42
HORIZONS = [1, 3, 7]
HOLIDAY_COUNTRY = "US"

# Chronological split aligned to TabTransformer: 70/15/15
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

#  ARTIFACTS 
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def make_run_dir_gbdt():
    ensure_dir("artifacts_gbdt")
    run_dir = os.path.join("artifacts_gbdt", datetime.now().strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_dir)
    latest = os.path.join("artifacts_gbdt", "latest_gbdt")
    try:
        if os.path.islink(latest): os.unlink(latest)
        elif os.path.exists(latest): os.remove(latest)
        os.symlink(os.path.abspath(run_dir), latest, target_is_directory=True)
    except Exception:
        with open(os.path.join("artifacts_gbdt","LATEST_PATH.txt"), "w") as f:
            f.write(os.path.abspath(run_dir))
    return run_dir

RUN_DIR = make_run_dir_gbdt()
print(f"[GBDT] Saving artifacts to: {RUN_DIR}")

#  OPTIONAL HOLIDAYS 
try:
    import holidays as _hol
    _holiday_calendar = _hol.CountryHoliday(HOLIDAY_COUNTRY)
    def is_holiday(d: pd.Timestamp) -> int: return int(d in _holiday_calendar)
    print(f"Holidays enabled for {HOLIDAY_COUNTRY}.")
except Exception:
    print("holidays package not found — proceeding without holiday flags.")
    def is_holiday(d: pd.Timestamp) -> int: return 0


print("Loading data...")
df = pd.read_csv(CSV_PATH)

#  NORMALIZE COLUMNS 
df = df.rename(columns={c: c.strip() for c in df.columns})
lower_cols = {c.lower(): c for c in df.columns}
def has(name): return name in lower_cols
def col(name): return lower_cols[name]

#  DATE/TIME PARSING 
if has("issue_datetime"):
    dt_series = pd.to_datetime(df[col("issue_datetime")], errors="coerce")
elif has("issue_date"):
    dt_series = pd.to_datetime(df[col("issue_date")], errors="coerce")
elif all(has(x) for x in ["year", "month", "day"]):
    dt_series = pd.to_datetime(
        df[col("year")].astype(str) + "-" +
        df[col("month")].astype(str) + "-" +
        df[col("day")].astype(str),
        errors="coerce"
    )
else:
    raise ValueError("No recognizable date field.")

df["_ISSUE_DATE"] = pd.to_datetime(pd.to_datetime(dt_series).dt.date)
df = df.dropna(subset=["_ISSUE_DATE"])

#  TARGET 
daily_counts = df.groupby("_ISSUE_DATE").size().rename("tickets").sort_index().astype(float)
y_log = np.log1p(daily_counts.values)
# roll-7 mean of log1p counts up to t-1 (same anchor as TabTransformer)
roll7_log = (
    pd.Series(y_log, index=daily_counts.index)
    .rolling(7, min_periods=1).mean().shift(1)
).fillna(method="bfill").astype(float)

#  FEATURE ENGINEERING (match tab transformer inputs at day-level) 
feat = pd.DataFrame(index=daily_counts.index)

def add_time_features(frame: pd.DataFrame, index_like: pd.DatetimeIndex) -> pd.DataFrame:
    f = frame.copy(); f.index = index_like
    f["day_of_week"] = index_like.dayofweek
    f["is_weekend"]  = (f["day_of_week"] >= 5).astype(int)
    f["is_holiday"]  = pd.Index(index_like).map(lambda d: is_holiday(pd.Timestamp(d))).astype(int)
    f["week_of_year"] = index_like.isocalendar().week.astype(int)
    f["month"] = index_like.month
    f["day_of_month"] = index_like.day
    f["days_since_start"] = (index_like - index_like.min()).days
    return f

X_base = add_time_features(feat, feat.index).sort_index()
X_base["roll7_log"] = roll7_log.reindex(X_base.index).astype(float)

y_base = daily_counts.reindex(X_base.index).astype(float)

#  METRICS 
def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.maximum(1, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
def wape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps)
def smape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))

def chrono_split_three(X, y, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    n = len(X)
    i_tr = int(np.floor(train_frac * n))
    i_va = int(np.floor((train_frac + val_frac) * n))
    return (X.iloc[:i_tr], X.iloc[i_tr:i_va], X.iloc[i_va:],
            y.iloc[:i_tr], y.iloc[i_tr:i_va], y.iloc[i_va:])

def residual_target(y_log_series: pd.Series, anchor_roll7_log: pd.Series, h: int) -> pd.Series:
    """r_h(t) = log1p(y_{t+h}) - roll7_log(t)"""
    y_future_log = pd.Series(np.r_[y_log_series.values[h:], np.full(h, np.nan)], index=y_log_series.index)
    return y_future_log - anchor_roll7_log

#  OVERALL COLLECTORS 
VAL_TRUE = {h: [] for h in HORIZONS}
VAL_PRED = {h: [] for h in HORIZONS}
TEST_TRUE = {h: [] for h in HORIZONS}
TEST_PRED = {h: [] for h in HORIZONS}
VAL_PER_H = {}
TEST_PER_H = {}

def per_h_metrics(yt, yp):
    return dict(
        mae=float(mean_absolute_error(yt, yp)),
        rmse=float(root_mean_squared_error(yt, yp)),
        smape=float(smape(yt, yp)),
        wape=float(wape(yt, yp)),
        r2=float(r2_score(yt, yp)) if len(yt) > 1 else None
    )

def compute_overall(y_true_dict, y_pred_dict, per_h=None):
    yt_all = np.concatenate([np.asarray(y_true_dict[h]) for h in y_true_dict])
    yp_all = np.concatenate([np.asarray(y_pred_dict[h]) for h in y_pred_dict])
    micro = dict(
        mae=float(mean_absolute_error(yt_all, yp_all)),
        rmse=float(root_mean_squared_error(yt_all, yp_all)),
        smape=float(smape(yt_all, yp_all)),
        wape=float(wape(yt_all, yp_all)),
        r2=float(r2_score(yt_all, yp_all))
    )
    results = {"micro": micro}
    if per_h:
        macro = {}
        for k in ["mae", "rmse", "smape", "wape", "r2"]:
            vals = [per_h[h][k] for h in per_h if per_h[h][k] is not None]
            macro[k] = float(np.mean(vals)) if len(vals) else None
        results["macro"] = macro
    return results

#  TRAIN/EVAL 
models_by_h = {}

def plot_feature_importance_barh(importances_mean, feature_names, title, out_path):
    try:
        idx = np.argsort(importances_mean)[::-1][:15]
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(feature_names)[idx][::-1], np.array(importances_mean)[idx][::-1])
        plt.title(title); plt.tight_layout()
        plt.savefig(out_path, dpi=150); plt.close()
    except Exception as e:
        print("[Warn] Could not plot feature importance:", e)

def train_eval_for_horizon(h, X, y, y_log_ser, roll7_log_ser):
    r_h = residual_target(y_log_ser, roll7_log_ser, h)
    mask = r_h.notna()
    X_h, r_h = X.loc[mask], r_h.loc[mask].astype(float)

    X_tr, X_va, X_te, r_tr, r_va, r_te = chrono_split_three(X_h, r_h)

    print(f"\n==== Horizon t+{h} (residual) ====")
    print(f"Train: {X_tr.index.min().date()} → {X_tr.index.max().date()}  "
          f"| Val: {X_va.index.min().date()} → {X_va.index.max().date()}  "
          f"| Test: {X_te.index.min().date()} → {X_te.index.max().date()}  "
          f"| N(tr)={len(X_tr)}, N(va)={len(X_va)}, N(te)={len(X_te)}")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_depth=None,
        max_iter=600,
        min_samples_leaf=10,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr, r_tr)
    print(f"Fitted trees (n_iter_): {getattr(model, 'n_iter_', 'NA')}")

    def recon(X_part):
        r_hat = model.predict(X_part)
        base  = X_part["roll7_log"].values
        y_hat = np.expm1(r_hat + base)
        y_true = y_base.shift(-h).reindex(X_part.index).values
        return y_true, y_hat

    y_tr_true, y_tr_pred = recon(X_tr)
    y_va_true, y_va_pred = recon(X_va)
    y_te_true, y_te_pred = recon(X_te)

    def report(split, yt, yp):
        print(f"  {split:<5} RMSE: {root_mean_squared_error(yt, yp):,.2f} | "
              f"MAE: {mean_absolute_error(yt, yp):,.2f} | "
              f"MAPE: {mape(yt, yp):,.2f}%")

    print("Performance (reconstructed counts):")
    report("Train", y_tr_true, y_tr_pred)
    report("Val",   y_va_true, y_va_pred)
    report("Test",  y_te_true, y_te_pred)

    # Save per-horizon predictions (val/test)
    pd.DataFrame({"date": X_va.index + pd.to_timedelta(h, unit="D"),
                  "y_true": y_va_true, "y_pred": y_va_pred}) \
      .to_csv(os.path.join(RUN_DIR, f"val_h{h}.csv"), index=False)

    pd.DataFrame({"date": X_te.index + pd.to_timedelta(h, unit="D"),
                  "y_true": y_te_true, "y_pred": y_te_pred}) \
      .to_csv(os.path.join(RUN_DIR, f"test_h{h}.csv"), index=False)

    try:
        perm = permutation_importance(model, X_va, r_va, n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
        plot_feature_importance_barh(
            perm.importances_mean, X_va.columns,
            title=f"Permutation Importance (residual r_h, t+{h})",
            out_path=os.path.join(RUN_DIR, f"feat_importance_h{h}.png")
        )
    except Exception as e:
        print("Permutation importance skipped:", e)

    VAL_TRUE[h] = y_va_true; VAL_PRED[h] = y_va_pred
    TEST_TRUE[h] = y_te_true; TEST_PRED[h] = y_te_pred
    VAL_PER_H[h] = per_h_metrics(y_va_true, y_va_pred)
    TEST_PER_H[h] = per_h_metrics(y_te_true, y_te_pred)

    models_by_h[h] = {
        "model": model,
        "train_index_min": X_tr.index.min(),
        "idx_splits": (X_tr.index, X_va.index, X_te.index)
    }

for h in HORIZONS:
    train_eval_for_horizon(h, X_base, y_base, pd.Series(y_log, index=y_base.index), roll7_log)

#  OVERALL METRICS 
VAL_OVERALL  = compute_overall(VAL_TRUE,  VAL_PRED,  per_h=VAL_PER_H)
TEST_OVERALL = compute_overall(TEST_TRUE, TEST_PRED, per_h=TEST_PER_H)

metrics_payload = {
    "val":  {"per_horizon": VAL_PER_H,  "overall": VAL_OVERALL},
    "test": {"per_horizon": TEST_PER_H, "overall": TEST_OVERALL}
}
with open(os.path.join(RUN_DIR, "gbdt_metrics.json"), "w") as f:
    json.dump(metrics_payload, f, indent=2)

def _fmt(x, is_pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "NA"
    return f"{x:.2f}%" if is_pct else (f"{x:.0f}" if abs(x) >= 100 else f"{x:.2f}")

rows = []
for split in ["val", "test"]:
    overall = metrics_payload[split]["overall"]
    for kind in ["micro", "macro"]:
        m = overall.get(kind, {})
        rows.append({
            "split": split, "avg": kind,
            "MAE": m.get("mae"), "RMSE": m.get("rmse"),
            "sMAPE(%)": m.get("smape"), "WAPE(%)": m.get("wape"),
            "R2": m.get("r2"),
        })
df_overall = pd.DataFrame(rows)
df_overall.to_csv(os.path.join(RUN_DIR, "gbdt_overall_metrics.csv"), index=False)

print("\n=== Overall Metrics (GBDT residualized) ===")
for _, r in df_overall.iterrows():
    print(
        f"{r['split'].upper()} {r['avg']:>5}: "
        f"MAE={_fmt(r['MAE'])} | RMSE={_fmt(r['RMSE'])} | "
        f"sMAPE={_fmt(r['sMAPE(%)'], True)} | WAPE={_fmt(r['WAPE(%)'], True)} | "
        f"R²={r['R2']:.3f}" if pd.notna(r['R2']) else
        f"{r['split'].upper()} {r['avg']:>5}: "
        f"MAE={_fmt(r['MAE'])} | RMSE={_fmt(r['RMSE'])} | "
        f"sMAPE={_fmt(r['sMAPE(%)'], True)} | WAPE={_fmt(r['WAPE(%)'], True)} | "
        f"R²=NA"
    )

#  PLOTTING 
def plot_pred_vs_actual(h, which="test", title_model_name="HGBR (residualized)", save_path=None):
    if h not in models_by_h:
        raise ValueError(f"No model for horizon {h}.")
    X_tr_idx, X_va_idx, X_te_idx = models_by_h[h]["idx_splits"]
    Xp_idx = X_va_idx if which == "val" else X_te_idx

    model = models_by_h[h]["model"]
    y_true = y_base.shift(-h).reindex(Xp_idx)
    r_hat  = model.predict(X_base.loc[Xp_idx])
    y_pred = np.expm1(r_hat + X_base.loc[Xp_idx, "roll7_log"].values)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Predicted vs actual daily violations (t+{h}, {which} split, {title_model_name}).",
                 fontsize=14, fontweight="bold")

    for day in y_true.index:
        if day.weekday() >= 5:
            plt.axvspan(day, day + pd.Timedelta(days=1), color="gray", alpha=0.12, linewidth=0)
    for day in y_true.index:
        if is_holiday(day):
            plt.axvline(day, color="red", linestyle=":", alpha=0.7, linewidth=1.5)

    plt.plot(y_true.index, y_true.values, label="Actual")
    plt.plot(y_true.index, y_pred, label="Predicted", linestyle="--")
    plt.title(f"Pred vs Actual ({title_model_name}) – t+{h} ({which})")
    plt.xlabel("Date"); plt.ylabel("Violations"); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path is None:
        save_path = os.path.join(RUN_DIR, f"{which}_pred_vs_actual_h{h}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

# Save plots for VAL/TEST
for h in HORIZONS:
    plot_pred_vs_actual(h, which="val")
    plot_pred_vs_actual(h, which="test")

#  SIMPLE FORECAST DEMO 
def _make_future_features(dates_idx: pd.DatetimeIndex, train_index_min: pd.Timestamp) -> pd.DataFrame:
    f = pd.DataFrame(index=dates_idx)
    f = add_time_features(f, dates_idx)
    # align & order columns
    for c in X_base.columns:
        if c not in f.columns:
            f[c] = 0
    f = f.reindex(columns=X_base.columns, fill_value=0)
    return f

def predict_for_dates(dates, horizon=1) -> pd.DataFrame:
    if horizon not in models_by_h:
        raise ValueError(f"No model for horizon {horizon}.")
    model = models_by_h[horizon]["model"]
    dates_idx = pd.to_datetime(pd.Index(dates)).normalize()
    Xf = _make_future_features(dates_idx, X_base.index.min())

    # roll7_log for future: fallback to last known anchor
    last_roll7 = float(X_base["roll7_log"].iloc[-1])
    Xf["roll7_log"] = last_roll7

    r_hat = model.predict(Xf)
    y_hat = np.expm1(r_hat + Xf["roll7_log"].values)
    target_dates = dates_idx + pd.to_timedelta(horizon, unit="D")
    return pd.DataFrame({"date": target_dates, f"predicted_tickets_t+{horizon}": y_hat.astype(float)}).set_index("date")

if __name__ == "__main__":
    future_dates = pd.date_range(start=X_base.index.max() + pd.Timedelta(days=1),
                                 periods=14, freq="D")
    for h in HORIZONS:
        preds = predict_for_dates(future_dates, horizon=h)
        preds.round(2).to_csv(os.path.join(RUN_DIR, f"forecast_next14_h{h}.csv"))
