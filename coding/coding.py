


%pip install -q --upgrade yfinance

# Setup
import os, sys, json, time, hashlib, platform, shutil, warnings, datetime as dt
from pathlib import Path
from importlib import metadata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=FutureWarning)


SEED        = 123
DATA_START  = "2022-01-01"
DATA_END    = "2026-11-01"
TEST_START  = "2024-01-01"

ENTRY_LAG   = 1
EXIT_LAG    = 20
TOP_N       = 10
WEIGHTING   = "pred_proportional"
PERIODS_PER_YEAR = 12
RF_MODE     = "tbill_proxy"

TREE_METHOD = "hist"
N_JOBS      = 1
N_ITER      = 40
CV_EDGES    = [0.40, 0.55, 0.70, 0.85, 1.00]
N_BOOT      = 10_000

USE_CACHE     = True
SAVE_TO_DRIVE = False


np.random.seed(SEED)

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path("/content/drive/MyDrive/upenn_xgb_project")
else:
    BASE_DIR = Path(os.environ.get("PROJECT_DIR", "/content/upenn_xgb_project"))

DATA_DIR, TABLE_DIR, FIG_DIR = BASE_DIR/"data", BASE_DIR/"outputs"/"tables", BASE_DIR/"outputs"/"figures"
for d in (DATA_DIR, TABLE_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _v(pkg):
    try: return metadata.version(pkg)
    except Exception: return "n/a"

VERSIONS = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "xgboost": getattr(xgb, "__version__", "n/a"), "yfinance": _v("yfinance")}
print("Versions:", VERSIONS)
print("Project folder:", BASE_DIR)
TEST_START_TS = pd.Timestamp(TEST_START)

# Stock universe
TICKERS = [

  "AAPL","MSFT","NVDA","GOOG","META","AMZN","AVGO","ORCL","ADBE","CRM",
  "CSCO","AMD","INTC","QCOM","TXN","IBM","NOW","INTU","AMAT","MU",

  "JPM","BAC","WFC","GS","MS","BLK","C","SCHW","SPGI","AXP",
  "USB","PNC","TFC","COF",

  "UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","DHR","ABT","BMY",
  "AMGN","GILD","CVS","ISRG","MDT",

  "TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG","ORLY",

  "WMT","COST","PG","KO","PEP","PM","MO","CL","EL",

  "CAT","GE","HON","UPS","BA","RTX","DE","LMT","MMM","ETN",

  "XOM","CVX","COP","SLB","EOG",

  "NFLX","DIS","CMCSA","TMUS","VZ",

  "AMT","PLD","NEE","DUK","SO","LIN","APD",
]
BENCHMARK = "SPY"
assert len(TICKERS) == len(set(TICKERS)), "duplicate tickers"
print(f"Tickers in the universe list: {len(TICKERS)}  (the R script's comment says 100)")

# Download prices
RAW_CSV, IRX_CSV, META_JSON = DATA_DIR/"raw_prices_long.csv", DATA_DIR/"tbill_irx.csv", DATA_DIR/"download_metadata.json"
FIELD_MAP = {"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adjusted","Volume":"volume"}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download_prices(symbols, start, end, retries=3):
    import yfinance as yf
    last = None
    for k in range(retries):
        try:
            raw = yf.download(list(symbols), start=start, end=end, auto_adjust=False,
                              actions=False, progress=False, threads=True)
            if raw is None or raw.empty:
                raise RuntimeError("Yahoo returned an empty frame")
            return raw
        except Exception as e:
            last = e; print(f"  download attempt {k+1} failed: {e}"); time.sleep(5 * (k + 1))
    raise RuntimeError(f"Could not download prices: {last}")

def to_long(raw, symbols):
    """Wide yfinance frame (MultiIndex columns) -> long frame: symbol,date,open,high,low,close,adjusted,volume."""
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({symbols[0]: raw}, axis=1).swaplevel(axis=1)
    lvl = 1 if set(symbols) & set(raw.columns.get_level_values(1)) else 0
    frames = []
    for s in symbols:
        if s not in set(raw.columns.get_level_values(lvl)):
            continue
        sub = raw.xs(s, axis=1, level=lvl)[list(FIELD_MAP)].rename(columns=FIELD_MAP)
        idx = pd.DatetimeIndex(sub.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        sub.index = idx.normalize(); sub.index.name = "date"
        sub = sub.reset_index(); sub.insert(0, "symbol", s)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)

symbols_all = TICKERS + [BENCHMARK]
if USE_CACHE and RAW_CSV.exists():
    long_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
    meta = json.loads(META_JSON.read_text()) if META_JSON.exists() else {}
    ok = (meta.get("raw_sha256") == sha256_file(RAW_CSV)) if meta else None
    print(f"Loaded cached snapshot {RAW_CSV.name}. Hash matches metadata: {ok}  (downloaded: {meta.get('downloaded_utc','unknown')})")
else:
    print("Downloading from Yahoo Finance ...")
    long_raw = to_long(download_prices(symbols_all, DATA_START, DATA_END), symbols_all)
    long_raw.to_csv(RAW_CSV, index=False)
    meta = {"downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "start": DATA_START, "end_exclusive": DATA_END, "versions": VERSIONS,
            "n_rows": int(len(long_raw)), "raw_sha256": sha256_file(RAW_CSV)}
    META_JSON.write_text(json.dumps(meta, indent=2))
    print("Saved snapshot:", RAW_CSV, "| sha256:", meta["raw_sha256"][:16], "...")

def load_tbill():
    """13-week T-bill yield (percent) from ^IRX; returns None if unavailable."""
    try:
        if USE_CACHE and IRX_CSV.exists():
            s = pd.read_csv(IRX_CSV, parse_dates=["date"]).set_index("date")["irx"]
        else:
            import yfinance as yf
            r = yf.download("^IRX", start=DATA_START, end=DATA_END, auto_adjust=False, progress=False)
            c = r["Close"]; c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            idx = pd.DatetimeIndex(c.index)
            if idx.tz is not None: idx = idx.tz_localize(None)
            s = pd.Series(c.to_numpy(dtype=float), index=idx.normalize(), name="irx").dropna()
            s.rename_axis("date").reset_index().to_csv(IRX_CSV, index=False)
        return s.sort_index() if len(s) else None
    except Exception as e:
        print("WARNING: could not load ^IRX; risk-free = 0.", e); return None

IRX = load_tbill() if RF_MODE == "tbill_proxy" else None
print("Risk-free series:", "T-bill proxy (^IRX)" if IRX is not None else "zero")

# Check data
PRICE_COLS = ["open","high","low","close","adjusted"]

def clean_universe(long_raw, tickers, bench):
    df = long_raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["adjusted"]).drop_duplicates(["symbol", "date"])
    spy = df[df["symbol"] == bench].sort_values("date")
    if spy.empty:
        raise RuntimeError("SPY missing from the download")
    cal = pd.DatetimeIndex(spy["date"])
    keep, log = [], []
    for s in tickers:
        sub = df[df["symbol"] == s]
        if sub.empty:
            log.append((s, "not returned by Yahoo")); continue
        gap = cal.difference(pd.DatetimeIndex(sub["date"]))
        if len(gap):
            log.append((s, f"missing {len(gap)} SPY-calendar trading days (first: {gap[0].date()})")); continue
        keep.append(s)
    prices = df[df["symbol"].isin(keep) & df["date"].isin(cal)].sort_values(["symbol", "date"]).reset_index(drop=True)
    assert (prices[PRICE_COLS] > 0).all().all(), "non-positive prices found"
    spy_df = spy[["date", "adjusted"]].rename(columns={"adjusted": "spy_price"}).reset_index(drop=True)
    return prices, spy_df, cal, keep, log

prices_long, spy_df, CAL, UNIVERSE, DROPPED = clean_universe(long_raw, TICKERS, BENCHMARK)

print(f"Universe used: {len(UNIVERSE)} of {len(TICKERS)} tickers")
if DROPPED:
    print("DROPPED tickers (deviation from the original universe):")
    for s, why in DROPPED: print("  ", s, "-", why)
print(f"Calendar: {CAL[0].date()} -> {CAL[-1].date()} | {len(CAL)} trading days")


chk = prices_long.assign(ret=prices_long.groupby("symbol")["adjusted"].pct_change(fill_method=None))
display(chk.reindex(chk["ret"].abs().sort_values(ascending=False).index).head(8)[["symbol", "date", "adjusted", "ret"]])

# Build features
FEATURES_ORIG = ["open","high","low","close","volume","adjusted",
                 "return","spy_ret","mom_1m","mom_3m","mom_6m","vol_1m","ret_5d","excess_ret"]
FEATURES_STATIONARY = ["return","spy_ret","mom_1m","mom_3m","mom_6m","vol_1m","ret_5d","excess_ret"]

def make_panel(long, spy):
    """long: symbol,date,open,high,low,close,adjusted,volume.  spy: date,spy_price."""
    spy = spy.sort_values("date").copy()
    spy["spy_ret"] = spy["spy_price"] / spy["spy_price"].shift(1) - 1

    df = long.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["return"] = df["adjusted"] / df.groupby("symbol", sort=False)["adjusted"].shift(1) - 1
    df = df.merge(spy[["date", "spy_ret"]], on="date", how="left")

    g = df.groupby("symbol", sort=False)
    adj = g["adjusted"]
    df["mom_1m"] = adj.shift(21)  / adj.shift(1) - 1
    df["mom_3m"] = adj.shift(63)  / adj.shift(1) - 1
    df["mom_6m"] = adj.shift(126) / adj.shift(1) - 1
    df["vol_1m"] = g["return"].transform(lambda s: s.shift(1).rolling(22, min_periods=22).std(ddof=1))
    df["ret_5d"] = adj.shift(5)   / adj.shift(1) - 1
    df["excess_ret"] = df["return"] - df["spy_ret"]
    df = df.dropna().reset_index(drop=True)

    g = df.groupby("symbol", sort=False)
    df["target"]     = g["adjusted"].shift(-EXIT_LAG) / g["adjusted"].shift(-ENTRY_LAG) - 1
    df["entry_date"] = g["date"].shift(-ENTRY_LAG)
    df["exit_date"]  = g["date"].shift(-EXIT_LAG)
    return df.dropna().reset_index(drop=True)

panel = make_panel(prices_long, spy_df)
print(f"Panel: {panel.shape[0]:,} rows | {panel['symbol'].nunique()} tickers | "
      f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
print("Predictors (14, as in the R recipe):", FEATURES_ORIG)
display(panel[["symbol","date","entry_date","exit_date","target"] + FEATURES_ORIG].head(3))

# Train/test split
train_orig = panel[panel["date"] <  TEST_START_TS].copy()
test       = panel[panel["date"] >= TEST_START_TS].copy()
train      = train_orig[train_orig["exit_date"] < TEST_START_TS].copy().reset_index(drop=True)
test       = test.reset_index(drop=True)

n_purged = len(train_orig) - len(train)
print(f"train (purged): {len(train):,} rows | {train['date'].min().date()} -> {train['date'].max().date()} | last label ends {train['exit_date'].max().date()}")
print(f"purged from R-style train: {n_purged:,} rows whose label window reaches {TEST_START} or later")
print(f"test:           {len(test):,} rows | {test['date'].min().date()} -> {test['date'].max().date()} | last label ends {test['exit_date'].max().date()}")

def leakage_tests():

    rng = np.random.default_rng(0)
    dates = np.sort(panel["date"].unique())
    cutoff = pd.Timestamp(dates[len(dates) // 2])
    pl, sp = prices_long.copy(), spy_df.copy()
    pl[PRICE_COLS + ["volume"]] = pl[PRICE_COLS + ["volume"]].astype(float)
    m = (pl["date"] > cutoff).to_numpy()
    pl.loc[m, PRICE_COLS + ["volume"]] = pl.loc[m, PRICE_COLS + ["volume"]].to_numpy() * rng.uniform(0.5, 1.5, size=(m.sum(), 1))
    ms = (sp["date"] > cutoff).to_numpy()
    sp.loc[ms, "spy_price"] = sp.loc[ms, "spy_price"].to_numpy() * rng.uniform(0.5, 1.5, size=ms.sum())
    p2 = make_panel(pl, sp)
    a = panel[panel["date"] <= cutoff].set_index(["symbol", "date"])
    b = p2[p2["date"] <= cutoff].set_index(["symbol", "date"]).reindex(a.index)
    assert b[FEATURES_ORIG].notna().all().all(), "perturbed panel is missing rows"
    for c in FEATURES_ORIG:
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True), f"feature {c} depends on the future!"
    near = a.index.get_level_values("date") >= pd.Timestamp(sorted(a.index.get_level_values("date").unique())[-10])
    assert (~np.isclose(a.loc[near, "target"].to_numpy(), b.loc[near, "target"].to_numpy())).any(), "positive control failed (targets should depend on the future)"

    assert train["exit_date"].max() < TEST_START_TS and train["date"].max() < TEST_START_TS
    assert test["date"].min() >= TEST_START_TS
    assert train["exit_date"].max() < test["date"].min(), "a training label overlaps the test period"

    wide = prices_long.pivot(index="date", columns="symbol", values="adjusted").reindex(CAL)
    samp = panel.sample(min(500, len(panel)), random_state=0)
    pos = CAL.get_indexer(samp["date"])
    t_chk = np.array([wide.iloc[p + EXIT_LAG][s] / wide.iloc[p + ENTRY_LAG][s] - 1 for p, s in zip(pos, samp["symbol"])])
    assert np.allclose(samp["target"].to_numpy(), t_chk, rtol=1e-9, atol=1e-12), "target mismatch"
    assert (samp["entry_date"].to_numpy() == CAL[pos + ENTRY_LAG].to_numpy()).all()
    assert (samp["exit_date"].to_numpy()  == CAL[pos + EXIT_LAG].to_numpy()).all()
    print("All leakage / alignment tests passed.")

leakage_tests()

# Baseline model
hist_mean_by_symbol = train.groupby("symbol")["target"].mean()
test["pred_hist_mean"] = test["symbol"].map(hist_mean_by_symbol)
assert test["pred_hist_mean"].notna().all()
print("Per-stock training means (19-day return): min {:.4f} | median {:.4f} | max {:.4f}".format(
      hist_mean_by_symbol.min(), hist_mean_by_symbol.median(), hist_mean_by_symbol.max()))

# Original XGBoost
ORIG_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, gamma=0.0, min_child_weight=10,
                   subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0)

def make_xgb(params):
    return XGBRegressor(objective="reg:squarederror", tree_method=TREE_METHOD, random_state=SEED,
                        n_jobs=N_JOBS, verbosity=0, **params)

model_orig = make_xgb(ORIG_PARAMS).fit(train[FEATURES_ORIG], train["target"])
test["pred_xgb_original"] = model_orig.predict(test[FEATURES_ORIG])
print("Fitted original XGBoost on", len(train), "rows.")

# Tune XGBoost
SEARCH_SPACE = {
    "max_depth":        [1, 2, 3],
    "n_estimators":     [50, 100, 200, 300],
    "learning_rate":    [0.01, 0.02, 0.05],
    "min_child_weight": [50, 100, 200, 400],
    "subsample":        [0.5, 0.7, 0.9],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "reg_lambda":       [1.0, 10.0, 50.0, 100.0],
    "reg_alpha":        [0.0, 0.1, 1.0],
}

def make_folds(df, edges):
    dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
    cuts = [int(round(e * len(dates))) for e in edges]
    folds = []
    for k in range(len(edges) - 1):
        vd = dates[cuts[k]:cuts[k + 1]]
        lo, hi = vd[0], vd[-1]
        va = ((df["date"] >= lo) & (df["date"] <= hi)).to_numpy()
        tr = ((df["date"] < lo) & (df["exit_date"] < lo)).to_numpy()
        assert df.loc[tr, "exit_date"].max() < lo, "fold leakage"
        folds.append((tr, va, lo, hi))
    return folds

FOLDS = make_folds(train, CV_EDGES)
for i, (tr, va, lo, hi) in enumerate(FOLDS, 1):
    print(f"fold {i}: train rows {tr.sum():>6,} | validate {lo.date()} -> {hi.date()} ({va.sum():,} rows)")

def cv_score(params, df, folds, features):
    rows = []
    for k, (tr, va, lo, hi) in enumerate(folds, 1):
        m = make_xgb(params).fit(df.loc[tr, features], df.loc[tr, "target"])
        p, y = m.predict(df.loc[va, features]), df.loc[va, "target"].to_numpy()
        hm = df.loc[va, "symbol"].map(df[tr].groupby("symbol")["target"].mean()).to_numpy()
        rows.append({"fold": k,
                     "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                     "rmse_hist_mean": float(np.sqrt(np.mean((y - hm) ** 2))),
                     "corr": float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan})
    return pd.DataFrame(rows)

def sample_configs(space, n, seed):
    rng = np.random.default_rng(seed); keys = list(space)
    n = min(n, int(np.prod([len(v) for v in space.values()])))
    seen, out = set(), []
    while len(out) < n:
        combo = tuple(int(rng.integers(len(space[k]))) for k in keys)
        if combo in seen: continue
        seen.add(combo); out.append({**{k: space[k][i] for k, i in zip(keys, combo)}, "gamma": 0.0})
    return out

CONFIGS = sample_configs(SEARCH_SPACE, N_ITER, SEED)
t0, rows = time.time(), []
for i, cfg in enumerate(CONFIGS):
    s = cv_score(cfg, train, FOLDS, FEATURES_ORIG)
    rows.append({**cfg, "val_rmse": s["rmse"].mean(), "val_rmse_sd": s["rmse"].std(ddof=1),
                 "val_corr": s["corr"].mean(), "val_rmse_hist_mean": s["rmse_hist_mean"].mean()})
    if (i + 1) % 10 == 0 or i + 1 == len(CONFIGS):
        print(f"  {i+1}/{len(CONFIGS)} configurations done ({time.time()-t0:.0f}s)")
cv_results = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)

cv_orig = cv_score(ORIG_PARAMS, train, FOLDS, FEATURES_ORIG)
cv_ref = pd.DataFrame({"model": ["Historical mean", "Original XGBoost (reference)", "Best regularized (selected)"],
                       "mean validation RMSE": [cv_orig["rmse_hist_mean"].mean(), cv_orig["rmse"].mean(), cv_results.loc[0, "val_rmse"]],
                       "mean validation corr": [np.nan, cv_orig["corr"].mean(), cv_results.loc[0, "val_corr"]]})
print("\nValidation summary (training period only):"); display(cv_ref)
print("Top 10 configurations by mean validation RMSE:")
display(cv_results.head(10))
if cv_results.loc[0, "val_rmse"] >= cv_orig["rmse"].mean():
    print("NOTE: the best regularized configuration did NOT beat the original in validation RMSE.")

HP_KEYS = list(SEARCH_SPACE) + ["gamma"]
BEST_PARAMS = {k: cv_results.loc[0, k] for k in HP_KEYS}
BEST_PARAMS = {k: (int(v) if k in ("max_depth", "n_estimators", "min_child_weight") else float(v)) for k, v in BEST_PARAMS.items()}
(TABLE_DIR / "selected_hyperparameters_LOCKED_before_test.json").write_text(json.dumps(BEST_PARAMS, indent=2))
print("\nSelected (locked before touching the test set):", BEST_PARAMS)

# Fit regularized model
model_reg = make_xgb(BEST_PARAMS).fit(train[FEATURES_ORIG], train["target"])
test["pred_xgb_regularized"] = model_reg.predict(test[FEATURES_ORIG])
print("Fitted regularized XGBoost on the full purged training set.")

hp_table = pd.DataFrame({"Parameter": HP_KEYS,
                         "Original": [ORIG_PARAMS[k] for k in HP_KEYS],
                         "Regularized (selected)": [BEST_PARAMS[k] for k in HP_KEYS]})
display(hp_table)

# Forecast evaluation
MODELS = {"Historical mean": "pred_hist_mean",
          "XGBoost (original)": "pred_xgb_original",
          "XGBoost (regularized)": "pred_xgb_regularized"}

pred_cols = list(MODELS.values())
assert test[pred_cols + ["target"]].notna().all().all()
print(f"All three models are scored on the same {len(test):,} test rows.")

def mean_rank_ic(df, col):
    ics = []
    for _, sub in df.groupby("date"):
        if sub[col].nunique() < 2: continue
        ics.append(sub[col].rank().corr(sub["target"].rank()))
    ics = np.array(ics, dtype=float)
    return (np.nanmean(ics), float(np.nanmean(ics > 0))) if len(ics) else (np.nan, np.nan)

def forecast_metrics(df, col, ref="pred_hist_mean"):
    y, p, h = df["target"].to_numpy(), df[col].to_numpy(), df[ref].to_numpy()
    varying = np.std(p) > 0
    ic, _ = mean_rank_ic(df, col)
    return {"N test obs": len(df),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "MAE": float(np.mean(np.abs(y - p))),
            "Correlation (Pearson)": float(np.corrcoef(p, y)[0, 1]) if varying else np.nan,
            "Spearman": float(pd.Series(p).rank().corr(pd.Series(y).rank())) if varying else np.nan,
            "Mean daily rank IC": ic,
            "OOS R2 vs hist. mean": float(1 - np.sum((y - p) ** 2) / np.sum((y - h) ** 2)),
            "Prediction std": float(np.std(p, ddof=1))}

forecast_tbl = pd.DataFrame({lab: forecast_metrics(test, col) for lab, col in MODELS.items()}).T
forecast_tbl.index.name = "Model"
display(forecast_tbl)

# Portfolio backtest
spy_px = spy_df.set_index("date")["spy_price"]

def month_rebalance_dates(preds):
    d = pd.Series(np.sort(preds["date"].unique()))
    return d.groupby(d.dt.to_period("M")).min().tolist()

def select_portfolio(day, col, top_n, weighting):

    eligible = day.loc[np.isfinite(day[col])].copy()
    if eligible.empty:
        raise ValueError(f"No finite predictions for {col} on {day['date'].iloc[0]}")
    top = eligible.nlargest(top_n, col).copy()
    pos = top[col].clip(lower=0).to_numpy(dtype=float)
    total = float(pos.sum())
    if weighting == "equal" or not np.isfinite(total) or total <= 1e-12:
        weights = np.full(len(top), 1.0 / len(top))
    else:
        weights = pos / total
        weights = weights / weights.sum()
    top["weight"] = weights
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(f"Invalid portfolio weights for {col} on {day['date'].iloc[0]}")
    return top

def backtest_all(preds, model_cols):
    rows, hold = [], []
    for rd in month_rebalance_dates(preds):
        day = preds[preds["date"] == rd]
        ent, ext = day["entry_date"].unique(), day["exit_date"].unique()
        assert len(ent) == 1 and len(ext) == 1, "stocks disagree on the holding window"
        ent, ext = pd.Timestamp(ent[0]), pd.Timestamp(ext[0])
        y = IRX.asof(ent) if IRX is not None else 0.0
        rf = 0.0 if pd.isna(y) else float(y) / 100.0 * (ext - ent).days / 365.0
        row = {"rebalance_date": rd, "entry_date": ent, "exit_date": ext, "rf_period": rf}
        for lab, col in model_cols.items():
            top = select_portfolio(day, col, TOP_N, WEIGHTING)
            row[lab] = float((top["weight"] * top["target"]).sum())
            h = top[["symbol", col, "weight", "target"]].rename(columns={col: "prediction", "target": "realized_return"})
            h.insert(0, "model", lab); h.insert(0, "rebalance_date", rd); hold.append(h)
        row["Equal-weight universe (reference)"] = float(day["target"].mean())
        row["SPY"] = float(spy_px.loc[ext] / spy_px.loc[ent] - 1)
        rows.append(row)
    pdf = pd.DataFrame(rows)
    assert (pdf["entry_date"].iloc[1:].to_numpy() >= pdf["exit_date"].iloc[:-1].to_numpy()).all(), "overlapping holding periods"
    return pdf, pd.concat(hold, ignore_index=True)


period_df, holdings = backtest_all(test, MODELS)
print(f"{len(period_df)} non-overlapping holding periods: first entry {period_df['entry_date'].iloc[0].date()}, last exit {period_df['exit_date'].iloc[-1].date()}")
display(period_df.head())

# Performance summary
def perf_table(pdf, labels, ppy=PERIODS_PER_YEAR):
    rf, spy = pdf["rf_period"].to_numpy(), pdf["SPY"].to_numpy()
    out = {}
    for lab in labels:
        r = pdf[lab].to_numpy(); n = len(r)
        curve = np.concatenate([[1.0], np.cumprod(1 + r)])
        ex = r - rf
        X = np.column_stack([np.ones(n), spy - rf])
        alpha, beta = np.linalg.lstsq(X, ex, rcond=None)[0]
        out[lab] = {"Periods": n,
                    "Cumulative return": curve[-1] - 1,
                    "Annualized return": curve[-1] ** (ppy / n) - 1,
                    "Annualized volatility": np.std(r, ddof=1) * np.sqrt(ppy),
                    "Sharpe (excess of T-bill)": ex.mean() / ex.std(ddof=1) * np.sqrt(ppy),
                    "Sharpe (rf = 0)": r.mean() / r.std(ddof=1) * np.sqrt(ppy),
                    "Max drawdown (period-end)": (curve / np.maximum.accumulate(curve) - 1).min(),
                    "Beta vs SPY": beta,
                    "Annualized alpha vs SPY": alpha * ppy,
                    "% periods beating SPY": np.nan if lab == "SPY" else float(np.mean(r > spy))}
    t = pd.DataFrame(out).T; t.index.name = "Strategy"
    return t

def active_table(pdf, labels, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED); n = len(pdf)
    idx = rng.integers(0, n, size=(n_boot, n)); rows = {}
    for lab in labels:
        a = pdf[lab].to_numpy() - pdf["SPY"].to_numpy()
        b = a[idx].mean(axis=1); lo, hi = np.percentile(b, [2.5, 97.5])
        rows[lab] = {"Mean active return / period": a.mean(), "Bootstrap 2.5%": lo, "Bootstrap 97.5%": hi}
    t = pd.DataFrame(rows).T; t.index.name = "Strategy"; return t

MAIN_LABELS = list(MODELS) + ["Equal-weight universe (reference)", "SPY"]
perf_tbl   = perf_table(period_df, MAIN_LABELS)
active_tbl = active_table(period_df, list(MODELS) + ["Equal-weight universe (reference)"])
print(f"Sample: entries {period_df['entry_date'].iloc[0].date()} -> exits {period_df['exit_date'].iloc[-1].date()} | {len(period_df)} periods")
display(perf_tbl); display(active_tbl)

# Charts
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
STYLE = {"Historical mean": dict(color="tab:gray"), "XGBoost (original)": dict(color="tab:orange"),
         "XGBoost (regularized)": dict(color="tab:blue"), "SPY": dict(color="black", lw=2.4),
         "Equal-weight universe (reference)": dict(color="tab:green", ls="--")}

def curve(pdf, col): return np.concatenate([[1.0], np.cumprod(1 + pdf[col].to_numpy())])
xdates = [period_df["entry_date"].iloc[0]] + list(period_df["exit_date"])


fig, ax = plt.subplots(figsize=(10, 5.5))
for lab in MAIN_LABELS:
    ax.plot(xdates, curve(period_df, lab), label=lab, **{"lw": 1.8, **STYLE[lab]})
ax.set(title="Growth of $1 over aligned holding periods (out-of-sample)", ylabel="Growth of $1", xlabel="Date")
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_cumulative_growth.png", dpi=200, bbox_inches="tight"); plt.show()


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (m, title) in zip(axes, [("RMSE", "RMSE (lower is better)"), ("MAE", "MAE (lower is better)"),
                                 ("Correlation (Pearson)", "Pooled correlation (higher is better)")]):
    vals = forecast_tbl[m]
    bars = ax.bar(range(len(vals)), vals.values, color=[STYLE[l]["color"] for l in vals.index])
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([l.replace(" (", "\n(") for l in vals.index], fontsize=8)
    ax.set_title(title, fontsize=10)
    for b, v in zip(bars, vals.values):
        if np.isfinite(v): ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
fig.suptitle("Out-of-sample forecast accuracy (identical test rows)", y=1.02); fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_forecast_metrics.png", dpi=200, bbox_inches="tight"); plt.show()


def gain_share(model, feats):
    g = model.get_booster().get_score(importance_type="gain")
    s = pd.Series({f: g.get(f, 0.0) for f in feats}); return s / s.sum() if s.sum() > 0 else s
imp = pd.DataFrame({"XGBoost (original)": gain_share(model_orig, FEATURES_ORIG),
                    "XGBoost (regularized)": gain_share(model_reg, FEATURES_ORIG)})
imp = imp.loc[imp.sum(axis=1).sort_values().index]
ax = imp.plot.barh(figsize=(9, 6), color=[STYLE[c]["color"] for c in imp.columns])
ax.set(title="Feature importance (share of total gain, fitted on training data)", xlabel="Share of gain")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig3_feature_importance.png", dpi=200, bbox_inches="tight"); plt.show()


fig, ax = plt.subplots(figsize=(10, 5))
spy_curve = curve(period_df, "SPY")
for lab in list(MODELS) + ["Equal-weight universe (reference)"]:
    ax.plot(xdates, curve(period_df, lab) / spy_curve, label=lab, **{"lw": 1.8, **STYLE[lab]})
ax.axhline(1, color="black", lw=1)
ax.set(title="Relative wealth vs SPY (above 1 = ahead of SPY)", ylabel="Strategy growth ÷ SPY growth", xlabel="Date")
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_relative_wealth_vs_spy.png", dpi=200, bbox_inches="tight"); plt.show()

# Zero-return benchmark
zero_pred = np.zeros(len(test), dtype=float)
y_actual = test["target"].to_numpy(dtype=float)
zero_metrics = {"N test obs": len(test),
                "RMSE": float(np.sqrt(np.mean((y_actual-zero_pred)**2))),
                "MAE": float(np.mean(np.abs(y_actual-zero_pred))),
                "Correlation (Pearson)": np.nan,
                "Spearman": np.nan,
                "Mean daily rank IC": np.nan,
                "OOS R2 vs hist. mean": float(1-np.sum(y_actual**2)/np.sum((y_actual-test["pred_hist_mean"].to_numpy())**2)),
                "Prediction std": 0.0}
forecast_with_zero = pd.concat([forecast_tbl, pd.DataFrame([zero_metrics], index=["Zero return"])])
display(forecast_with_zero[["N test obs","RMSE","MAE","Correlation (Pearson)","OOS R2 vs hist. mean"]])


# Transaction costs
COST_PER_SIDE = 0.001
cost_labels = MAIN_LABELS
period_after_cost = period_df.copy()
for label in cost_labels:
    period_after_cost[label] = (1-COST_PER_SIDE)**2 * (1+period_df[label]) - 1

net_perf_tbl = perf_table(period_after_cost, cost_labels)
cost_comparison = pd.DataFrame({
    "Gross cumulative return": perf_tbl.loc[cost_labels, "Cumulative return"],
    "Net cumulative return (0.1% per side)": net_perf_tbl.loc[cost_labels, "Cumulative return"],
    "Gross Sharpe (excess T-bill)": perf_tbl.loc[cost_labels, "Sharpe (excess of T-bill)"],
    "Net Sharpe (excess T-bill)": net_perf_tbl.loc[cost_labels, "Sharpe (excess of T-bill)"],
    "Number of round trips": len(period_df),
})
display(cost_comparison)


TABLE_DIR.mkdir(parents=True, exist_ok=True)
forecast_with_zero.to_csv(TABLE_DIR / "table9_forecast_with_zero.csv")
cost_comparison.to_csv(TABLE_DIR / "table10_cost_sensitivity.csv")
net_perf_tbl.to_csv(TABLE_DIR / "table11_net_performance.csv")
period_after_cost.to_csv(TABLE_DIR / "period_returns_after_cost.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
for label in ["Historical mean", "XGBoost (original)", "XGBoost (regularized)", "SPY"]:
    growth = np.r_[1.0, np.cumprod(1+period_after_cost[label].to_numpy())]
    dates = [period_after_cost["entry_date"].iloc[0]] + period_after_cost["exit_date"].tolist()
    ax.plot(dates, growth, label=label)
ax.set(title="Growth of $1 after illustrative round-trip costs", xlabel="Date", ylabel="Growth of $1")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_cumulative_growth_after_cost.png", dpi=160)
plt.show()


# Additional results
m_unpurged = make_xgb(ORIG_PARAMS).fit(train_orig[FEATURES_ORIG], train_orig["target"])
test["pred_xgb_original_unpurged"] = m_unpurged.predict(test[FEATURES_ORIG])

m_stat = make_xgb(BEST_PARAMS).fit(train[FEATURES_STATIONARY], train["target"])
test["pred_xgb_regularized_stationary"] = m_stat.predict(test[FEATURES_STATIONARY])

SUPP = {"XGBoost (original, un-purged train = R split)": "pred_xgb_original_unpurged",
        "XGBoost (regularized, stationary features only)": "pred_xgb_regularized_stationary"}

supp_forecast = pd.DataFrame({lab: forecast_metrics(test, col) for lab, col in SUPP.items()}).T
supp_forecast.index.name = "Model"

all_models = {**MODELS, **SUPP}
period_all, holdings_all = backtest_all(test, all_models)
assert (period_all["entry_date"].to_numpy() == period_df["entry_date"].to_numpy()).all()
supp_perf = perf_table(period_all, list(SUPP) + ["SPY"])
print("Supplementary forecast metrics:"); display(supp_forecast)
print("Supplementary portfolio metrics:"); display(supp_perf)

# Export tables
def best(series, hi=True, fmt="{:.4f}"):
    s = series.dropna(); k = s.idxmax() if hi else s.idxmin(); return f"{k} ({fmt.format(s[k])})"

pt = perf_tbl.loc[list(MODELS) + ["SPY"]]
print(f"Test rows: {len(test):,} | holding periods: {len(period_df)} | sample: "
      f"{period_df['entry_date'].iloc[0].date()} -> {period_df['exit_date'].iloc[-1].date()}")
print("Lowest RMSE:                ", best(forecast_tbl["RMSE"], hi=False))
print("Lowest MAE:                 ", best(forecast_tbl["MAE"], hi=False))
print("Highest pooled correlation: ", best(forecast_tbl["Correlation (Pearson)"]))
print("Highest mean rank IC:       ", best(forecast_tbl["Mean daily rank IC"]))
print("Highest annualized return:  ", best(pt["Annualized return"], fmt="{:.2%}"))
print("Highest Sharpe (excess):    ", best(pt["Sharpe (excess of T-bill)"], fmt="{:.2f}"))
for lab in MODELS:
    print(f"{lab:<24} cumulative {perf_tbl.loc[lab,'Cumulative return']:+.2%} vs SPY {perf_tbl.loc['SPY','Cumulative return']:+.2%}"
          f" | active/period 95% bootstrap interval [{active_tbl.loc[lab,'Bootstrap 2.5%']:+.2%}, {active_tbl.loc[lab,'Bootstrap 97.5%']:+.2%}]")
print("\nReminder: ~{} periods and overlapping, cross-correlated labels mean differences may be within noise.".format(len(period_df)))

# Save outputs
def save_numeric(df, name, index=True):
    df.to_csv(TABLE_DIR / f"{name}.csv", index=index)

def save_md(df, name):
    try:    (TABLE_DIR / f"{name}.md").write_text(df.to_markdown(index=True))
    except Exception: (TABLE_DIR / f"{name}.html").write_text(df.to_html())

pct = lambda x: "–" if pd.isna(x) else f"{x:.2%}"
def fmt_forecast(t):
    f = t.copy()
    for c in ["RMSE", "MAE", "Prediction std"]: f[c] = f[c].map(lambda v: f"{v*100:.3f}%")
    for c in ["Correlation (Pearson)", "Spearman", "Mean daily rank IC", "OOS R2 vs hist. mean"]: f[c] = f[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    f["N test obs"] = f["N test obs"].map(lambda v: f"{int(v):,}"); return f
def fmt_perf(t):
    f = t.copy()
    for c in ["Cumulative return", "Annualized return", "Annualized volatility", "Max drawdown (period-end)", "Annualized alpha vs SPY", "% periods beating SPY"]:
        f[c] = f[c].map(pct)
    for c in ["Sharpe (excess of T-bill)", "Sharpe (rf = 0)", "Beta vs SPY"]: f[c] = f[c].map(lambda v: f"{v:.2f}")
    f["Periods"] = f["Periods"].astype(int); return f

save_numeric(forecast_tbl, "table1_forecast_accuracy");        save_md(fmt_forecast(forecast_tbl), "table1_forecast_accuracy_formatted")
save_numeric(perf_tbl,     "table2_portfolio_performance");    save_md(fmt_perf(perf_tbl),         "table2_portfolio_performance_formatted")
save_numeric(active_tbl,   "table3_active_return_vs_spy");     save_md(active_tbl.map(pct),        "table3_active_return_vs_spy_formatted")
save_numeric(hp_table,     "table4_hyperparameters", index=False)
save_numeric(cv_ref,       "table5_validation_summary", index=False)
save_numeric(cv_results,   "table6_validation_all_configs", index=False)
save_numeric(supp_forecast,"table7_supp_forecast");            save_numeric(supp_perf, "table8_supp_performance")
save_md(fmt_forecast(supp_forecast), "table7_supp_forecast_formatted"); save_md(fmt_perf(supp_perf), "table8_supp_performance_formatted")
save_numeric(period_all,   "period_returns_all", index=False)
save_numeric(holdings_all, "holdings_all_models", index=False)


methods = f"""# Methods, data, assumptions and limitations

## Data
* Source: Yahoo Finance daily prices via `yfinance` {VERSIONS['yfinance']} (`auto_adjust=False`, adjusted close = split- and dividend-adjusted). Benchmark: SPY. Risk-free proxy: ^IRX (13-week T-bill yield, annualized %), {'used' if IRX is not None else 'NOT available, risk-free set to 0'}.
* Download window: {DATA_START} to {DATA_END} (exclusive); snapshot downloaded {meta.get('downloaded_utc', 'unknown')}; file hash in `data/download_metadata.json`.
* Universe: {len(UNIVERSE)} of {len(TICKERS)} listed tickers (same list as the professor's script). Dropped: {', '.join(s for s, _ in DROPPED) or 'none'}.
* Feature/label panel: {len(panel):,} stock-days, {panel['date'].min().date()} to {panel['date'].max().date()}.

## Sample
* Training (purged): {train['date'].min().date()} to {train['date'].max().date()} ({len(train):,} rows; labels end by {train['exit_date'].max().date()}).
* Test: {test['date'].min().date()} to {test['date'].max().date()} ({len(test):,} rows).
* Backtest: {len(period_df)} monthly holding periods, entries from {period_df['entry_date'].iloc[0].date()} to exits by {period_df['exit_date'].iloc[-1].date()}.

## Design choices
* Target: return from close t+{ENTRY_LAG} to close t+{EXIT_LAG} (= {EXIT_LAG-ENTRY_LAG} daily returns), as in the R script.
* Leakage control: training rows whose label window reaches {TEST_START} are removed (purge); validation folds are walk-forward and purged the same way; hyper-parameters were fixed before any test prediction.
* Models: historical per-stock mean; original XGBoost (300 trees, depth 5, lr 0.05, min_child_weight 10, gamma 0, parsnip defaults assumed for the rest); regularized XGBoost chosen by lowest mean validation RMSE over {len(CONFIGS)} random configurations: {BEST_PARAMS}.
* Portfolio: first trading day of each month; top {TOP_N} by prediction; weight proportional to max(prediction, 0); buy at close t+{ENTRY_LAG}, sell at close t+{EXIT_LAG}. SPY measured over identical dates.
* Annualization: {PERIODS_PER_YEAR} periods per year; idle days between holding periods are treated as 0% cash for every series.

## Differences from the R implementation
Python/yfinance instead of R/tidyquant; frozen end date and cached snapshot; label purge at the train/test boundary; SPY aligned to the portfolio's holding windows (R used calendar months); 12 (not 252) periods per year for Sharpe/alpha; optional T-bill risk-free; XGBoost library/version defaults (`tree_method`={TREE_METHOD}, `base_score`) may differ from the R build, so predictions are not bit-identical.

## Limitations
* **Survivorship / hindsight universe:** today's large caps (several are recent winners) were chosen with knowledge of the outcome; results are not representative of a real-time selection.
* **Short, single-regime sample** with few non-overlapping periods; Sharpe ratios and active-return estimates are very noisy. The bootstrap interval assumes i.i.d. periods.
* **Overlapping labels** across adjacent days and strong cross-sectional correlation make pooled RMSE/MAE/correlation far less informative than the row count suggests; no formal significance tests were run.
* **Original feature set** includes non-stationary raw price levels and volume and unconventional "momentum" ratios (kept for fidelity).
* **Gross results exclude costs.** A supplementary sensitivity scenario assumes 0.1% per side, full liquidation/re-entry for every period; this does not model variable slippage, taxes, market impact or capacity constraints; fills at closing prices.
* **Static model:** trained once, never re-fit during the test period (the original design); a production system would retrain on a rolling basis.
* **Data:** Yahoo adjusted prices can be revised; corporate actions (e.g. spin-offs) may be handled imperfectly; the T-bill proxy uses a simple-interest approximation.
* Hyper-parameter search is small and random; the selected configuration may be fragile. The XGBoost hyper-parameter and feature-set choices were not tuned on the test period, but the sensitivity checks in section 11 were examined after seeing test results and should be read as diagnostics.
"""
(BASE_DIR / "outputs" / "methods_and_limitations.md").write_text(methods)

run_meta = {"run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "versions": VERSIONS,
            "config": {k: globals()[k] for k in ["SEED","DATA_START","DATA_END","TEST_START","ENTRY_LAG","EXIT_LAG","TOP_N","WEIGHTING",
                                                "PERIODS_PER_YEAR","RF_MODE","TREE_METHOD","N_JOBS","N_ITER","CV_EDGES","N_BOOT"]},
            "data": {**{k: meta.get(k) for k in ("downloaded_utc", "raw_sha256")}, "universe_used": UNIVERSE, "dropped": DROPPED,
                     "rf_series": "IRX" if IRX is not None else "zero"},
            "rows": {"panel": len(panel), "train_orig": len(train_orig), "train_purged": len(train), "test": len(test)},
            "original_params": ORIG_PARAMS, "selected_params": BEST_PARAMS}
(BASE_DIR / "outputs" / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, default=str))

archive = shutil.make_archive(str(BASE_DIR / "blog_outputs"), "zip", BASE_DIR / "outputs")
print("Saved to:", BASE_DIR / "outputs"); print("Zip:", archive)
for p in sorted((BASE_DIR / "outputs").rglob("*")):
    if p.is_file(): print("  ", p.relative_to(BASE_DIR))
try:
    from google.colab import files; files.download(archive)
except Exception:
    pass
display(Markdown(methods))
