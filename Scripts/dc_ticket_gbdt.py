# gbdt_daily_parking_violations_multi_horizon_holidays.py
# Python 3.9+; pandas, numpy, scikit-learn; optional: holidays
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance

# ---------- CONFIG ----------
CSV_PATH = "../CleanData"   # folder containing CSVs
TOP_AGENCIES = 10
TOP_VIOLATIONS = 20
TEST_FRACTION = 0.3
RANDOM_STATE = 42
HORIZONS = [1, 3, 7]
HOLIDAY_COUNTRY = "US"      # e.g., "US", "CA", "GB", etc.

# ---------- OPTIONAL HOLIDAYS ----------
try:
    import holidays as _hol
    _holiday_calendar = _hol.CountryHoliday(HOLIDAY_COUNTRY)
    def is_holiday(d: pd.Timestamp) -> int:
        return int(d in _holiday_calendar)
    print(f"Holidays enabled for {HOLIDAY_COUNTRY}.")
except Exception as e:
    print("holidays package not found or failed to init — proceeding without holiday flags.")
    def is_holiday(d: pd.Timestamp) -> int:
        return 0

# ---------- LOAD ----------
def load_parking_violation_data(data_folder):
    all_csvs = glob.glob(os.path.join(data_folder, "*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")
    dfs = []
    for c in all_csvs:
        try:
            df = pd.read_csv(c)
            dfs.append(df)
            print(f"loaded: {c}")
        except Exception as e:
            print(f"error opening {c}: {e}")
    return pd.concat(dfs, ignore_index=True)

print("Loading data...")
df = load_parking_violation_data(CSV_PATH)

# ---------- NORMALIZE COLUMNS ----------
cols = {c: c.strip() for c in df.columns}
df = df.rename(columns=cols)
lower_cols = {c.lower(): c for c in df.columns}
def has(name): return name in lower_cols
def col(name): return lower_cols[name]

# ---------- DATE/TIME PARSING ----------
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
    raise ValueError("Could not find a date field (issue_datetime, issue_date, or year/month/day).")

df["_ISSUE_DATE"] = pd.to_datetime(pd.to_datetime(dt_series).dt.date)
df = df.dropna(subset=["_ISSUE_DATE"])

# ---------- TARGET ----------
daily_counts = df.groupby("_ISSUE_DATE").size().rename("tickets").sort_index()

# ---------- FEATURE ENGINEERING ----------
feat = pd.DataFrame(index=daily_counts.index)

def add_time_features(frame: pd.DataFrame, index_like: pd.DatetimeIndex) -> pd.DataFrame:
    f = frame.copy()
    f.index = index_like
    f["day_of_week"] = index_like.dayofweek                 # 0=Mon
    f["is_weekend"]  = (f["day_of_week"] >= 5).astype(int)
    # holiday flag (uses optional holidays lib; otherwise returns 0)
    f["is_holiday"]  = pd.Index(index_like).map(lambda d: is_holiday(pd.Timestamp(d))).astype(int)
    # basic calendar nums
    f["week_of_year"] = index_like.isocalendar().week.astype(int)
    f["month"] = index_like.month
    f["day_of_month"] = index_like.day
    return f

feat = add_time_features(feat, feat.index)
feat["days_since_start"] = (feat.index - feat.index.min()).days

# Agency features
agency_col = None
for candidate in ["issuing_agency_short", "issuing_agency_name", "issuing_agency_code"]:
    if has(candidate):
        agency_col = col(candidate); break

if agency_col is not None:
    top_agencies = (
        df[agency_col].astype(str).fillna("UNKNOWN").value_counts().head(TOP_AGENCIES).index
    )
    df["_agency"] = df[agency_col].astype(str).fillna("UNKNOWN").where(
        df[agency_col].isin(top_agencies), other="OTHER"
    )
    agency_pivot = pd.crosstab(df["_ISSUE_DATE"], df["_agency"])
    keep = [c for c in agency_pivot.columns if (c in top_agencies) or (c == "OTHER")]
    agency_pivot = agency_pivot[keep].reindex(feat.index, fill_value=0).add_prefix("agency_")
    feat = feat.join(agency_pivot)

# Violation features
viol_col = col("violation_code") if has("violation_code") else None
if viol_col is not None:
    top_viol = df[viol_col].astype(str).fillna("UNKNOWN").value_counts().head(TOP_VIOLATIONS).index
    df["_viol"] = df[viol_col].astype(str).fillna("UNKNOWN").where(
        df[viol_col].isin(top_viol), other="OTHER"
    )
    viol_pivot = pd.crosstab(df["_ISSUE_DATE"], df["_viol"])
    keep = [c for c in viol_pivot.columns if (c in top_viol) or (c == "OTHER")]
    viol_pivot = viol_pivot[keep].reindex(feat.index, fill_value=0).add_prefix("viol_")
    feat = feat.join(viol_pivot)

# Base matrices
X_base = feat.sort_index()
y_base = daily_counts.reindex(X_base.index).astype(int)

# ---------- UTILITIES ----------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.maximum(1, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def time_split(X, y, test_fraction=TEST_FRACTION):
    n = len(X)
    test_size = max(1, int(np.floor(test_fraction * n)))
    split_idx = n - test_size
    return (X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:])

# ---------- TRAINING (MULTI-HORIZON) ----------
models_by_h = {}

def train_eval_for_horizon(h, X, y):
    # Align target at t+h
    y_h = y.shift(-h)
    mask = y_h.notna()
    X_h, y_h = X.loc[mask], y_h.loc[mask].astype(int)

    X_tr, X_te, y_tr, y_te = time_split(X_h, y_h, TEST_FRACTION)

    print(f"\n==== Horizon t+{h} ====")
    print(f"Train: {X_tr.index.min().date()} → {X_tr.index.max().date()}  "
          f"| Test: {X_te.index.min().date()} → {X_te.index.max().date()}  "
          f"| N(train)={len(X_tr)}, N(test)={len(X_te)}")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_depth=None,
        max_iter=600,
        min_samples_leaf=10,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)
    print(f"Fitted trees (n_iter_): {getattr(model, 'n_iter_', 'NA')}")

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    print("Performance:")
    print(f"  Train RMSE: {root_mean_squared_error(y_tr, pred_tr):,.2f} | "
          f"MAE: {mean_absolute_error(y_tr, pred_tr):,.2f} | "
          f"MAPE: {mape(y_tr, pred_tr):,.2f}%")
    print(f"   Test RMSE: {root_mean_squared_error(y_te, pred_te):,.2f} | "
          f"MAE: {mean_absolute_error(y_te, pred_te):,.2f} | "
          f"MAPE: {mape(y_te, pred_te):,.2f}%")

    # Optional permutation importance
    try:
        perm = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=RANDOM_STATE)
        imp = (pd.DataFrame({"feature": X_te.columns, "importance": perm.importances_mean})
                 .sort_values("importance", ascending=False)
                 .head(15))
        print("\nTop features (permutation importance):")
        for _, r in imp.iterrows():
            print(f"  {r['feature']}: {r['importance']:.4f}")
    except Exception as e:
        print("Permutation importance skipped:", e)

    return model, X_h.index

for h in HORIZONS:
    m, idx = train_eval_for_horizon(h, X_base, y_base)
    models_by_h[h] = {"model": m, "train_index_min": idx.min()}

# ---------- PREDICTION ----------
def _make_future_features(dates_idx: pd.DatetimeIndex, train_index_min: pd.Timestamp) -> pd.DataFrame:
    f = pd.DataFrame(index=dates_idx)
    f = add_time_features(f, dates_idx)
    f["days_since_start"] = (dates_idx - train_index_min).days
    # fill missing engineered columns with 0 and align column order to training matrix
    for c in X_base.columns:
        if c not in f.columns:
            f[c] = 0
    return f.reindex(columns=X_base.columns, fill_value=0)

def predict_for_dates(dates, horizon=1) -> pd.DataFrame:
    if horizon not in models_by_h:
        raise ValueError(f"No model for horizon {horizon}. Available: {list(models_by_h.keys())}")
    model = models_by_h[horizon]["model"]
    tmin  = models_by_h[horizon]["train_index_min"]

    dates_idx = pd.to_datetime(pd.Index(dates)).normalize()
    Xf = _make_future_features(dates_idx, tmin)
    preds = model.predict(Xf)

    target_dates = dates_idx + pd.to_timedelta(horizon, unit="D")
    return pd.DataFrame(
        {"date": target_dates, f"predicted_tickets_t+{horizon}": preds.astype(float)}
    ).set_index("date")

# ---------- PLOTTING ----------
def _recreate_test_split_for_h(h):
    y_h = y_base.shift(-h)
    mask = y_h.notna()
    X_h, y_h = X_base.loc[mask], y_h.loc[mask].astype(int)
    return time_split(X_h, y_h, TEST_FRACTION)

def plot_pred_vs_actual(h, title_model_name="HistGradientBoostingRegressor", save_path=None):
    """Plot test window with weekends shaded and holidays marked."""
    if h not in models_by_h:
        raise ValueError(f"No model for horizon {h}.")
    _, X_test, _, y_test = _recreate_test_split_for_h(h)
    model = models_by_h[h]["model"]
    y_pred = model.predict(X_test)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Figure: Predicted vs. actual daily violations (t+{h} horizon, {title_model_name}).",
                 fontsize=14, fontweight="bold")

    # Weekend shading
    for day in y_test.index:
        if day.weekday() >= 5:
            plt.axvspan(day, day + pd.Timedelta(days=1), color="gray", alpha=0.12, linewidth=0)

    # Holiday markers
    for day in y_test.index:
        if is_holiday(day):
            plt.axvline(day, color="red", linestyle=":", alpha=0.7, linewidth=1.5)

    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, y_pred, label="Predicted", linestyle="--")
    plt.title(f"Pred vs Actual ({title_model_name}) – t+{h}")
    plt.xlabel("Date"); plt.ylabel("Violations"); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# ---------- EXAMPLES ----------
if __name__ == "__main__":
    # Plot test-set fit for each horizon (with weekend/holiday highlights)
    plot_pred_vs_actual(1)
    plot_pred_vs_actual(3)
    plot_pred_vs_actual(7)

    # Forecast next 14 days at each horizon (weekend/holiday features included)
    future_feature_dates = pd.date_range(start=X_base.index.max() + pd.Timedelta(days=1),
                                         periods=14, freq="D")
    for h in HORIZONS:
        preds = predict_for_dates(future_feature_dates, horizon=h)
        print(f"\nForecast (t+{h}) next 14 days:")
        print(preds.round(2))

