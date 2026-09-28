import numpy as np
import pandas as pd

def get_feed_values(df):
    """
    Extracts glucose and xylose concentrations from the feed sample row.
    Assumes only one experiment label per DataFrame.
    """
    required_cols = {"Experiment Label", "Sample name", "Glucose", "Xylose"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing one of the required columns: {required_cols}")

    exp_label = df["Experiment Label"].dropna().unique()
    if len(exp_label) != 1:
        raise ValueError(f"Expected exactly one unique Experiment Label, found: {exp_label}")

    label = exp_label[0]

    # Two naming conventions appear in the workbook:
    #   BC/VR runs  -> "<LABEL>_M", "<LABEL>_1", ..., "<LABEL>_Feed"
    #   M runs      -> "M", "1", ..., "F"
    # Try the labelled form first, then the bare "F"/"Feed" form.
    names = df["Sample name"].astype(str).str.strip()
    feed_row = df[names == f"{label}_Feed"]
    if feed_row.empty:
        feed_row = df[names.isin(["F", "Feed"])]
    if feed_row.empty:
        raise ValueError(
            f"No feed row for experiment {label}: expected Sample name "
            f"'{label}_Feed' or 'F'/'Feed'. Found: {sorted(names.unique())}"
        )

    glucose = float(feed_row["Glucose"].values[0])
    xylose = float(feed_row["Xylose"].values[0])

    return glucose, xylose


def create_feed_rate_function(df):
    """
    Returns a function f(t) that gives the stepwise feed rate [mL/h] at any time t,
    based on experimental data. Assumes backward step behavior.
    """
    df = df.copy()

    # Ensure numeric columns
    df["Time (h)"] = pd.to_numeric(df["Time (h)"], errors="coerce")
    df["Feed rate (mL/h)"] = pd.to_numeric(df["Feed rate (mL/h)"], errors="coerce")
    df["Feed added (mL)"] = pd.to_numeric(df["Feed added (mL)"], errors="coerce")

    # Case 1: All feed rates are NaN → return zero function
    if df["Feed rate (mL/h)"].isna().all():
        def zero_feed_rate(_t):
            return 0.0
        return zero_feed_rate

    # Case 2: Valid feed rates exist → build stepwise function
    df = df.dropna(subset=["Time (h)", "Feed rate (mL/h)"]).reset_index(drop=True)

    # Infer actual feed start time
    t1 = df.loc[0, "Time (h)"]
    feed_added_1 = df.loc[0, "Feed added (mL)"]
    feed_rate_1 = df.loc[0, "Feed rate (mL/h)"]
    t_feed_start = t1 - (feed_added_1 / feed_rate_1)

    # Define stepwise time series
    time_points = np.array([t_feed_start] + df["Time (h)"].tolist())
    feed_rates = np.array(df["Feed rate (mL/h)"].tolist())

    def feed_rate_at_time(t):
        if t < time_points[0]:
            return 0.0
        idx = int(np.searchsorted(time_points, t, side="right")) - 1
        # time_points has one more entry than feed_rates (it is prefixed with
        # the inferred feed start), so idx == len(feed_rates) whenever t lands
        # on or past the final breakpoint. Clamp to the last segment: the feed
        # is held at its final rate to the end of the run.
        if idx >= len(feed_rates):
            idx = len(feed_rates) - 1
        return float(feed_rates[idx])

    return feed_rate_at_time



# ─────────────────────────────────────────────────────────────────────────────
#  Cached accessor — use this inside ODE right-hand sides
# ─────────────────────────────────────────────────────────────────────────────

def get_feed_cached(exter):
    """Return (S1_feed, S2_feed, feed_func) for `exter`, computed once.

    `get_feed_values` and `create_feed_rate_function` both do substantial pandas
    work (DataFrame copies, to_numeric coercion, dropna, boolean masking). They
    were previously called on EVERY evaluation of the ODE right-hand side, where
    they accounted for over 99% of the runtime -- roughly 770 us per call against
    3 us for the kinetics themselves, a 235x overhead.

    The feed profile and feed concentrations are properties of the experiment and
    cannot change during an integration, so they are computed once and cached on
    the DataFrame object. A DataFrame built by `df.copy()` is a different object
    and gets its own cache, so derived frames never inherit a stale profile.
    """
    cached = getattr(exter, "_biokin_feed_cache", None)
    if cached is None:
        s1, s2 = get_feed_values(exter)
        cached = (s1, s2, create_feed_rate_function(exter))
        try:
            object.__setattr__(exter, "_biokin_feed_cache", cached)
        except Exception:
            pass  # un-cacheable object: correct, just slow
    return cached
