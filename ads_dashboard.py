
# ====== STREAMLIT DASHBOARD APP ======
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Ad Performance Dashboard", page_icon="📊", layout="wide")

def find_col_contains(candidates, columns):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for cand in candidates:
        for c in columns:
            if cand.lower() in c.lower():
                return c
    return None

def normalize_rate(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mx = np.nanmax(s.values) if np.any(~np.isnan(s.values)) else np.nan
    if not np.isnan(mx) and mx > 1.5:
        s = s / 100.0
    return s.clip(0, 1)

MONTH_MAP = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
WEEKDAY_MAP = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

ECPI = "expected_conv_per_impression"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def ensure_time_features(df: pd.DataFrame, ts_col: str | None) -> pd.DataFrame:
    out = df.copy()
    if ts_col and ts_col in out.columns:
        ts = pd.to_datetime(out[ts_col], errors="coerce")
        if "hour" not in out.columns: out["hour"] = ts.dt.hour
        if "dayofweek" not in out.columns: out["dayofweek"] = ts.dt.dayofweek
        if "month" not in out.columns: out["month"] = ts.dt.month
        if "is_weekend" not in out.columns: out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return out

def add_bins(df: pd.DataFrame, col_age=None, col_income=None) -> pd.DataFrame:
    out = df.copy()
    if col_age and col_age in out.columns:
        age = pd.to_numeric(out[col_age], errors="coerce")
        out["AgeGroup"] = pd.cut(age, bins=[0, 29, 49, 200], labels=["Young", "Middle", "Elder"])
    if col_income and col_income in out.columns:
        inc = pd.to_numeric(out[col_income], errors="coerce")
        try:
            out["IncomeGroup"] = pd.qcut(inc, q=3, labels=["Low", "Medium", "High"])
        except Exception:
            out["IncomeGroup"] = pd.cut(inc, bins=3, labels=["Low", "Medium", "High"])
    if "dayofweek" in out.columns:
        out["Weekday"] = pd.to_numeric(out["dayofweek"], errors="coerce").map(WEEKDAY_MAP)
    if "month" in out.columns:
        m = pd.to_numeric(out["month"], errors="coerce")
        out["MonthName"] = m.dropna().astype(int).map(MONTH_MAP)
    return out

def plot_mean_ecpi_by_category(df, cat_col, title, top_k=20, min_count=30, height=420):
    if cat_col is None or cat_col not in df.columns:
        st.info(f"Skip: {title} (column not found)")
        return
    tmp = df[[cat_col, ECPI]].dropna()
    if tmp.empty:
        st.info(f"Skip: {title} (no data after filtering)")
        return
    counts = tmp[cat_col].value_counts()
    keep = counts.head(top_k).index
    tmp = tmp[tmp[cat_col].isin(keep)]
    grp = tmp.groupby(cat_col)[ECPI].agg(mean="mean", count="count").reset_index()
    grp = grp[grp["count"] >= min_count].sort_values("mean", ascending=False)
    if grp.empty:
        st.info(f"Skip: {title} (no categories with count ≥ {min_count})")
        return
    fig = px.bar(grp, x=cat_col, y="mean", text="mean", title=title, hover_data={"mean":":.6f", "count":True})
    fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
    fig.update_layout(height=height, xaxis_tickangle=-35, yaxis_title="Mean ECPI")
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Mean ECPI: %{y:.6f}<br>Count: %{customdata[1]}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

st.title("📊 Advertising Performance Dashboard")
st.caption("Interactive ECPI analytics with filters (Streamlit + Plotly)")

DATA_PATH = st.sidebar.text_input("Dataset path", value="Dataset_Ads.csv")
df_raw = load_data(DATA_PATH)

col_ctr = find_col_contains(["CTR","click_through_rate","clickthroughrate"], df_raw.columns)
col_conv = find_col_contains(["Conversion Rate","conversion_rate","ConversionRate"], df_raw.columns)
col_ts = find_col_contains(["Timestamp","time","datetime","date"], df_raw.columns)

col_placement = find_col_contains(["Ad Placement","placement","ad_placement"], df_raw.columns)
col_topic     = find_col_contains(["Ad Topic","Topic","AdTopic"], df_raw.columns)
col_type      = find_col_contains(["Ad Type","Type","AdType"], df_raw.columns)
col_location  = find_col_contains(["Location","country","region","city"], df_raw.columns)
col_gender    = find_col_contains(["Gender","sex"], df_raw.columns)
col_age       = find_col_contains(["Age"], df_raw.columns)
col_income    = find_col_contains(["Income"], df_raw.columns)

df = ensure_time_features(df_raw, col_ts)

if ECPI not in df.columns:
    if col_ctr is None or col_conv is None:
        st.error(f"ECPI '{ECPI}' not found and CTR/Conv not detected.")
        st.stop()
    df[col_ctr] = normalize_rate(df[col_ctr])
    df[col_conv] = normalize_rate(df[col_conv])
    df[ECPI] = (df[col_ctr] * df[col_conv]).clip(0, 1)

df = add_bins(df, col_age=col_age, col_income=col_income)

st.sidebar.header("🔎 Filters")

def multiselect_filter(label, df, col):
    if col and col in df.columns:
        opts = sorted(df[col].dropna().astype(str).unique().tolist())
        return st.sidebar.multiselect(label, opts, default=opts)
    return None

sel_placement = multiselect_filter("Ad Placement", df, col_placement)
sel_topic     = multiselect_filter("Ad Topic", df, col_topic)
sel_type      = multiselect_filter("Ad Type", df, col_type)
sel_location  = multiselect_filter("Location", df, col_location)
sel_gender    = multiselect_filter("Gender", df, col_gender)

sel_weekday = st.sidebar.multiselect("Weekday", WEEKDAY_ORDER, default=WEEKDAY_ORDER) if "Weekday" in df.columns else None
sel_month   = st.sidebar.multiselect("Month", MONTH_ORDER, default=MONTH_ORDER) if "MonthName" in df.columns else None

min_count = st.sidebar.slider("Min category count (bar charts)", 5, 100, 30, 5)
top_k     = st.sidebar.slider("Top categories shown", 5, 40, 20, 1)

df_f = df.copy()

def apply_in_filter(df_in, col, selected):
    if col and selected is not None:
        return df_in[df_in[col].astype(str).isin(set(map(str, selected)))]
    return df_in

df_f = apply_in_filter(df_f, col_placement, sel_placement)
df_f = apply_in_filter(df_f, col_topic, sel_topic)
df_f = apply_in_filter(df_f, col_type, sel_type)
df_f = apply_in_filter(df_f, col_location, sel_location)
df_f = apply_in_filter(df_f, col_gender, sel_gender)

if sel_weekday is not None and "Weekday" in df_f.columns:
    df_f = df_f[df_f["Weekday"].astype(str).isin(sel_weekday)]
if sel_month is not None and "MonthName" in df_f.columns:
    df_f = df_f[df_f["MonthName"].astype(str).isin(sel_month)]

df_f = df_f.dropna(subset=[ECPI])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df_f):,}")
c2.metric("Mean ECPI", f"{df_f[ECPI].mean():.6f}" if len(df_f) else "—")
c3.metric("Median ECPI", f"{df_f[ECPI].median():.6f}" if len(df_f) else "—")
c4.metric("Unique Placements", f"{df_f[col_placement].nunique():,}" if col_placement else "—")

st.divider()

left, right = st.columns([1.1, 1.0])
with left:
    fig = px.histogram(df_f, x=ECPI, nbins=40, title="ECPI Distribution", hover_data={ECPI:":.6f"})
    fig.update_layout(height=380, xaxis_title="ECPI", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with right:
    num_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Correlation Heatmap")
        fig.update_layout(height=380)
        fig.update_xaxes(tickangle=90)
        fig.update_traces(hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Corr: %{z:.4f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

a, b = st.columns(2)
with a:
    plot_mean_ecpi_by_category(df_f, col_placement, "Mean ECPI by Placement", top_k=top_k, min_count=min_count)
with b:
    plot_mean_ecpi_by_category(df_f, col_topic, "Mean ECPI by Topic", top_k=top_k, min_count=min_count)

with st.expander("📄 Preview filtered rows"):
    st.dataframe(df_f.head(200), use_container_width=True)
# ====== END STREAMLIT DASHBOARD APP ======
