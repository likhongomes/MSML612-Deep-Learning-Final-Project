# gbdt_daily_parking_violations_multi_horizon.py
# Requires: Python 3.9+, pandas, numpy, scikit-learn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = "../CleanData"  # <- your uploaded file
TOP_AGENCIES = 10
TOP_VIOLATIONS = 20
TEST_FRACTION = 0.3
RANDOM_STATE = 42
HORIZONS = [1, 3, 7]  # evaluate t+1, t+3, t+7

# ---------- LOAD ----------
print("Loading data...")

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

df = load_parking_violation_data(CSV_PATH)

# Normalize columns for easier matching
cols = {c: c.strip() for c in df.columns}
df = df.rename(columns=cols)
lower_cols = {c.lower(): c for c in df.columns}

def has(col): return col in lower_cols
def col(col): return lower_cols[col]

# Identify date column (issue_datetime preferred, else issue_date, else from year/month/day)
date_series = None
if has("issue_datetime"):
    date_series = pd.to_datetime(df[col("issue_datetime")], errors="coerce")
elif has("issue_date"):
    date_series = pd.to_datetime(df[col("issue_date")], errors="coerce")
elif all(has(x) for x in ["year", "month", "day"]):
    date_series = pd.to_datetime(
        df[col("year")].astype(str) + "-" +
        df[col("month")].astype(str) + "-" +
        df[col("day")].astype(str),
        errors="coerce"
    )
else:
    raise ValueError("Could not find a date field (issue_datetime, issue_date, or year/month/day).")

df["_ISSUE_DATE"] = pd.to_datetime(pd.to_datetime(date_series).dt.date)  # normalize to midnight
df = df.dropna(subset=["_ISSUE_DATE"])

# ---------- TARGET: daily counts ----------
daily_counts = df.groupby("_ISSUE_DATE").size().rename("tickets").sort_index()

# ---------- FEATURE ENGINEERING ----------
feat = pd.DataFrame(index=daily_counts.index)
feat["day_of_week"] = feat.index.dayofweek  # 0=Mon
feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(int)
feat["week_of_year"] = feat.index.isocalendar().week.astype(int)
feat["month"] = feat.index.month
feat["day_of_month"] = feat.index.day
feat["days_since_start"] = (feat.index - feat.index.min()).days

# Agency count features (same-day aggregates; if using for real forecasting, consider lagging these)
agency_col = None
for candidate in ["issuing_agency_short", "issuing_agency_name", "issuing_agency_code"]:
    if has(candidate):
        agency_col = col(candidate)
        break

if agency_col is not None:
    top_agencies = (
        df[agency_col].astype(str).fillna("UNKNOWN").value_counts().head(TOP_AGENCIES).index
    )
    df["_agency"] = df[agency_col].astype(str).fillna("UNKNOWN").where(
        df[agency_col].isin(top_agencies), other="OTHER"
    )
    agency_pivot = pd.crosstab(df["_ISSUE_DATE"], df["_agency"])
    keep_cols = [c for c in agency_pivot.columns if (c in top_agencies) or (c == "OTHER")]
    agency_pivot = agency_pivot[keep_cols].reindex(feat.index, fill_value=0)
    agency_pivot = agency_pivot.add_prefix("agency_")
    feat = feat.join(agency_pivot)

# Violation count features
viol_col = col("violation_code") if has("violation_code") else None
if viol_col is not None:
    top_viol = df[viol_col].astype(str).fillna("UNKNOWN").value_counts().head(TOP_VIOLATIONS).index
    df["_viol"] = df[viol_col].astype(str).fillna("UNKNOWN").where(
        df[viol_col].isin(top_viol), other="OTHER"
    )
    viol_pivot = pd.crosstab(df["_ISSUE_DATE"], df["_viol"])
    keep_cols = [c for c in viol_pivot.columns if (c in top_viol) or (c == "OTHER")]
    viol_pivot = viol_pivot[keep_cols].reindex(feat.index, fill_value=0)
    viol_pivot = viol_pivot.add_prefix("viol_")
    feat = feat.join(viol_pivot)

# Base feature/target
X_base = feat.sort_index()
y_base = daily_counts.reindex(X_base.index).astype(int)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(1, np.abs(y_true))  # avoid div/0
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def train_eval_for_horizon(h, X, y):
    """
    Align features at time t to predict target at t+h by shifting y up by h (negative shift).
    Drops the last h rows (no target available).
    """
    # Align target for horizon h
    y_h = y.shift(-h)
    # Keep only rows where target is available
    valid_mask = y_h.notna()
    X_h = X.loc[valid_mask].copy()
    y_h = y_h.loc[valid_mask].astype(int)

    # Time-ordered split
    n_days = len(X_h)
    test_size = max(1, int(np.floor(TEST_FRACTION * n_days)))
    split_idx = n_days - test_size
    X_train, X_test = X_h.iloc[:split_idx], X_h.iloc[split_idx:]
    y_train, y_test = y_h.iloc[:split_idx], y_h.iloc[split_idx:]

    print(f"\n==== Horizon t+{h} days ====")
    print(f"Days total={n_days}, train={len(X_train)}, test={len(X_test)}")
    print("Train date range:", X_train.index.min().date(), "to", X_train.index.max().date())
    print("Test  date range:", X_test.index.min().date(), "to", X_test.index.max().date())

    # Model
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_depth=None,
        max_iter=600,
        min_samples_leaf=10,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Evaluation
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    rmse_train = root_mean_squared_error(y_train, pred_train)
    rmse_test  = root_mean_squared_error(y_test,  pred_test)
    mae_train  = mean_absolute_error(y_train, pred_train)
    mae_test   = mean_absolute_error(y_test,  pred_test)
    mape_train = mape(y_train, pred_train)
    mape_test_ = mape(y_test,  pred_test)

    print("\n=== Performance ===")
    print(f"Train RMSE: {rmse_train:,.2f} | MAE: {mae_train:,.2f} | MAPE: {mape_train:,.2f}%")
    print(f" Test RMSE: {rmse_test:,.2f} | MAE: {mae_test:,.2f} | MAPE: {mape_test_:,.2f}%")

    # Permutation importance (optional)
    try:
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
        imp = (
            pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean})
            .sort_values("importance", ascending=False)
            .head(20)
        )
        print("\nTop features (permutation importance):")
        for _, row in imp.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        print("Permutation importance skipped:", e)

    return model, X_h.index, (X_train, X_test, y_train, y_test)

# ---------- RUN MULTI-HORIZON EVALUATIONS ----------
models_by_h = {}
for h in HORIZONS:
    model_h, index_h, _splits = train_eval_for_horizon(h, X_base, y_base)
    models_by_h[h] = {"model": model_h, "train_index_min": index_h.min(), "full_index": index_h}

# ---------- PREDICTOR FOR ARBITRARY DATES (uses last-fit model for each horizon) ----------
def _make_calendar_features(dates_idx: pd.DatetimeIndex, train_index_min: pd.Timestamp) -> pd.DataFrame:
    f = pd.DataFrame(index=dates_idx)
    f["day_of_week"] = f.index.dayofweek
    f["is_weekend"] = (f["day_of_week"] >= 5).astype(int)
    f["week_of_year"] = f.index.isocalendar().week.astype(int)
    f["month"] = f.index.month
    f["day_of_month"] = f.index.day
    f["days_since_start"] = (f.index - train_index_min).days
    return f

def predict_for_dates(dates, horizon=1) -> pd.DataFrame:
    """
    dates: list-like of 'YYYY-MM-DD' or pd.DatetimeIndex (these are the feature dates t)
    horizon: predict tickets at t+horizon
    Returns DataFrame with predictions indexed by the *target* date (t + horizon).
    """
    if horizon not in models_by_h:
        raise ValueError(f"No fitted model for horizon {horizon}. Available: {list(models_by_h.keys())}")

    model = models_by_h[horizon]["model"]
    train_index_min = models_by_h[horizon]["train_index_min"]

    dates_idx = pd.to_datetime(pd.Index(dates)).normalize()
    Xf = _make_calendar_features(dates_idx, train_index_min)

    # add zero columns for agency_*/viol_* seen in training
    for c in X_base.columns:
        if c not in Xf.columns:
            Xf[c] = 0
    Xf = Xf.reindex(columns=X_base.columns, fill_value=0)

    preds = model.predict(Xf)
    # shift index forward by horizon to represent the date being predicted
    target_dates = (dates_idx + pd.to_timedelta(horizon, unit="D"))
    out = pd.DataFrame({"date": target_dates, f"predicted_tickets_t+{horizon}": preds.astype(float)}).set_index("date")
    return out

# ---------- EXAMPLE USAGE ----------
# Predict for the next 7 calendar days (features at t), at each horizon
example_feature_dates = pd.date_range(start=(X_base.index.max() + pd.Timedelta(days=1)), periods=7, freq="D")
for h in HORIZONS:
    future_preds_h = predict_for_dates(example_feature_dates, horizon=h)
    print(f"\nPredictions for next 7 days (reported at target dates) for horizon t+{h}:")
    print(future_preds_h.round(2))





plt.figure(figsize=(12,6))

future_dates = pd.date_range(start=X_base.index.max() + pd.Timedelta(days=1),
                             periods=30, freq="D")

for h in [1,3,7]:
    preds = predict_for_dates(future_dates, horizon=h)
    plt.plot(preds.index, preds.iloc[:,0], marker="o", label=f"Horizon t+{h}")

plt.xlabel("Date")
plt.ylabel("Predicted tickets")
plt.title("Multi-horizon forecast")
plt.legend()
plt.tight_layout()
plt.show()






def _test_split_for_horizon(h, X, y, test_fraction=TEST_FRACTION):
    """Recreate the exact time-ordered split used for training at horizon h."""
    y_h = y.shift(-h)
    mask = y_h.notna()
    X_h, y_h = X.loc[mask], y_h.loc[mask].astype(int)

    n = len(X_h)
    test_size = max(1, int(np.floor(test_fraction * n)))
    split_idx = n - test_size
    X_train, X_test = X_h.iloc[:split_idx], X_h.iloc[split_idx:]
    y_train, y_test = y_h.iloc[:split_idx], y_h.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

def plot_pred_vs_actual(h, title_model_name="HistGradientBoostingRegressor", save_path=None):
    """Plot actual vs predicted on the test window for horizon h (t+h)."""
    if h not in models_by_h:
        raise ValueError(f"No model for horizon {h}. Have: {list(models_by_h.keys())}")

    # Recreate the test split and predict with the already-fitted model
    _, X_test, _, y_test = _test_split_for_horizon(h, X_base, y_base, TEST_FRACTION)
    model = models_by_h[h]["model"]
    y_pred = model.predict(X_test)

    # Figure caption (top) + plot title (inside)
    plt.figure(figsize=(12, 5.5))
    plt.suptitle(f"Figure: Predicted vs. actual daily violations (t+{h} horizon, {title_model_name}).", fontsize=14, fontweight="bold")
    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, y_pred, label="Predicted", linestyle="--")
    plt.title(f"Pred vs Actual ({title_model_name}) – t+{h}")
    plt.xlabel("Date")
    plt.ylabel("Violations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave space for the caption

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# --- Examples ---
plot_pred_vs_actual(1, title_model_name="HistGradientBoostingRegressor")   # t+1
plot_pred_vs_actual(3, title_model_name="HistGradientBoostingRegressor")   # t+3
plot_pred_vs_actual(7, title_model_name="HistGradientBoostingRegressor")   # t+7