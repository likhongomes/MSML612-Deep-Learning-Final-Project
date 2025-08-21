# gbdt_daily_parking_violations_untitled.py
# Requires: Python 3.9+, pandas, numpy, scikit-learn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.inspection import permutation_importance
import glob
import os

# ---------- CONFIG ----------
CSV_PATH = "../CleanData"  # <- your uploaded file
TOP_AGENCIES = 10
TOP_VIOLATIONS = 20
TEST_FRACTION = 0.3
RANDOM_STATE = 42

# ---------- LOAD ----------
print("Loading data...")
# df = pd.read_csv(CSV_PATH, low_memory=False)

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
    # parse as datetime, then normalize to date
    date_series = pd.to_datetime(df[col("issue_datetime")], errors="coerce")
elif has("issue_date"):
    date_series = pd.to_datetime(df[col("issue_date")], errors="coerce")
elif all(has(x) for x in ["year", "month", "day"]):
    # build from parts
    date_series = pd.to_datetime(
        df[col("year")].astype(str) + "-" +
        df[col("month")].astype(str) + "-" +
        df[col("day")].astype(str),
        errors="coerce"
    )
else:
    raise ValueError("Could not find a date field (issue_datetime, issue_date, or year/month/day).")

df["_ISSUE_DATE"] = pd.to_datetime(date_series.dt.date)  # normalize to midnight
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

# Agency count features
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

# ---------- TRAIN / TEST SPLIT (time-ordered) ----------
X = feat.sort_index()
y = daily_counts.reindex(X.index).astype(int)

n_days = len(X)
test_size = max(1, int(np.floor(TEST_FRACTION * n_days)))
split_idx = n_days - test_size
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Days total={n_days}, train={len(X_train)}, test={len(X_test)}")
print("Train date range:", X_train.index.min().date(), "to", X_train.index.max().date())
print("Test  date range:", X_test.index.min().date(), "to", X_test.index.max().date())

# ---------- MODEL ----------
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

# ---------- EVALUATION ----------
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(1, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

rmse_train = root_mean_squared_error(y_train, pred_train)
rmse_test  = root_mean_squared_error(y_test,  pred_test)
mae_train  = mean_absolute_error(y_train, pred_train)
mae_test   = mean_absolute_error(y_test,  pred_test)
mape_test  = mape(y_test, pred_test)

print("\n=== Performance ===")
print(f"Train RMSE: {rmse_train:,.2f} | MAE: {mae_train:,.2f}")
print(f" Test RMSE: {rmse_test:,.2f} | MAE: {mae_test:,.2f} | MAPE: {mape_test:,.2f}%")

# ---------- PERMUTATION IMPORTANCE (optional) ----------
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

# ---------- PREDICTOR FOR ARBITRARY DATES ----------
def _make_calendar_features(dates_idx: pd.DatetimeIndex, train_index: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dates_idx)
    f["day_of_week"] = f.index.dayofweek
    f["is_weekend"] = (f["day_of_week"] >= 5).astype(int)
    f["week_of_year"] = f.index.isocalendar().week.astype(int)
    f["month"] = f.index.month
    f["day_of_month"] = f.index.day
    f["days_since_start"] = (f.index - train_index.min()).days
    return f

def predict_for_dates(dates) -> pd.DataFrame:
    """
    dates: list-like of 'YYYY-MM-DD' or pd.DatetimeIndex
    Returns DataFrame with predictions.
    """
    dates_idx = pd.to_datetime(pd.Index(dates)).normalize()

    Xf = _make_calendar_features(dates_idx, X.index)

    # add zero columns for agency_*/viol_* seen in training
    for c in X.columns:
        if c not in Xf.columns:
            Xf[c] = 0

    Xf = Xf.reindex(columns=X.columns, fill_value=0)

    preds = model.predict(Xf)
    out = pd.DataFrame({"date": dates_idx, "predicted_tickets": preds.astype(float)}).set_index("date")
    return out

# ---------- EXAMPLE USAGE ----------
example_dates = pd.date_range(start=(X.index.max() + pd.Timedelta(days=1)), periods=7, freq="D")
future_preds = predict_for_dates(example_dates)
print("\nPredictions for next 7 days after dataset:")
print(future_preds.round(2))

# --- Metrics ---
# Mean Absolute Error
mae_train = mean_absolute_error(y_train, pred_train)
mae_test  = mean_absolute_error(y_test, pred_test)

# Root Mean Squared Error
rmse_train = root_mean_squared_error(y_train, pred_train)
rmse_test  = root_mean_squared_error(y_test, pred_test)

# Mean Absolute Percentage Error (manual implementation)
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.maximum(1, np.abs(y_true))  # avoid div/0
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

mape_train = mape(y_train, pred_train)
mape_test  = mape(y_test,  pred_test)

# --- Print ---
print("\n=== Performance ===")
print(f"Train MAE : {mae_train:,.2f}")
print(f" Test MAE : {mae_test:,.2f}")
print(f"Train RMSE: {rmse_train:,.2f}")
print(f" Test RMSE: {rmse_test:,.2f}")
print(f"Train MAPE: {mape_train:,.2f}%")
print(f" Test MAPE: {mape_test:,.2f}%")