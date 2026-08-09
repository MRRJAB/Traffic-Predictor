"""
Traffic Forecast Dashboard
----------------------------
Displays historical traffic data (Junction 1) and generates forecasts for
upcoming hours using the LightGBM model trained earlier in the pipeline.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Traffic Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLS = [
    'hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month',
    'hour_sin', 'hour_cos',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
    'rolling_mean_3h', 'rolling_mean_24h', 'rolling_mean_7d'
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
PRIMARY = "#3E6BE0"
ACCENT = "#F0A63A"
BG = "#0F1420"
PANEL = "#171D2E"
TEXT = "#E7E9EE"
SUBTEXT = "#8B93A7"
BORDER = "#252C40"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
        background-color: {PANEL};
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT} !important;
    }}
    h1, h2, h3 {{
        font-family: 'Georgia', serif;
        letter-spacing: -0.01em;
    }}
    .dash-title {{
        font-family: 'Georgia', serif;
        font-size: 2.1rem;
        font-weight: 600;
        color: {TEXT};
        margin-bottom: 0;
    }}
    .dash-subtitle {{
        color: {SUBTEXT};
        font-size: 0.95rem;
        margin-top: 4px;
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background-color: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 18px 22px;
        height: 100%;
    }}
    .metric-label {{
        color: {SUBTEXT};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .metric-value {{
        color: {TEXT};
        font-size: 1.9rem;
        font-weight: 600;
        font-family: 'Georgia', serif;
    }}
    .metric-sub {{
        color: {SUBTEXT};
        font-size: 0.82rem;
        margin-top: 4px;
    }}
    .status-banner {{
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.92rem;
        border: 1px solid;
        margin-top: 4px;
    }}
    .status-alert {{
        background-color: rgba(240, 166, 58, 0.08);
        border-color: rgba(240, 166, 58, 0.35);
        color: #F0A63A;
    }}
    .status-clear {{
        background-color: rgba(62, 107, 224, 0.08);
        border-color: rgba(62, 107, 224, 0.35);
        color: #7C9CF0;
    }}
    .section-label {{
        color: {SUBTEXT};
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.8rem;
        margin-bottom: 0.6rem;
    }}
    [data-testid="stExpander"] {{
        background-color: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("traffic_model_2026.pkl")


@st.cache_data
def load_history():
    df = pd.read_csv("traffic_2026_clean.csv")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df.sort_values('DateTime').reset_index(drop=True)


def build_features_for_timestamp(ts, series):
    """
    Builds the feature vector for a given timestamp, pulling past values
    from `series` (a dict of DateTime -> Vehicles containing both real
    history and previously generated forecasts).
    """
    def get(ts_lookup):
        return series.get(ts_lookup, np.nan)

    lag_1h = get(ts - pd.Timedelta(hours=1))
    lag_2h = get(ts - pd.Timedelta(hours=2))
    lag_3h = get(ts - pd.Timedelta(hours=3))
    lag_24h = get(ts - pd.Timedelta(hours=24))
    lag_168h = get(ts - pd.Timedelta(hours=168))

    last_3h = [get(ts - pd.Timedelta(hours=h)) for h in range(1, 4)]
    last_24h = [get(ts - pd.Timedelta(hours=h)) for h in range(1, 25)]
    last_7d = [get(ts - pd.Timedelta(hours=h)) for h in range(1, 24 * 7 + 1)]

    return {
        'hour': ts.hour,
        'day_of_week': ts.dayofweek,
        'is_weekend': int(ts.dayofweek >= 5),
        'day_of_month': ts.day,
        'month': ts.month,
        'hour_sin': np.sin(2 * np.pi * ts.hour / 24),
        'hour_cos': np.cos(2 * np.pi * ts.hour / 24),
        'lag_1h': lag_1h, 'lag_2h': lag_2h, 'lag_3h': lag_3h,
        'lag_24h': lag_24h, 'lag_168h': lag_168h,
        'rolling_mean_3h': np.nanmean(last_3h),
        'rolling_mean_24h': np.nanmean(last_24h),
        'rolling_mean_7d': np.nanmean(last_7d),
    }


def forecast_next_hours(model, history_df, start_ts, n_hours):
    """
    Recursive forecasting: each prediction is fed back into the series to
    serve as a lag feature for subsequent predictions, since the real
    future values beyond start_ts are unknown.
    """
    series = dict(zip(history_df['DateTime'], history_df['Vehicles']))
    predictions = []

    for i in range(1, n_hours + 1):
        ts = start_ts + pd.Timedelta(hours=i)
        feats = build_features_for_timestamp(ts, series)
        X = pd.DataFrame([feats])[FEATURE_COLS]
        pred = model.predict(X)[0]
        pred = max(0, pred)
        series[ts] = pred
        predictions.append({'DateTime': ts, 'prediction': pred})

    return pd.DataFrame(predictions)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="dash-title">Traffic Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-subtitle">Hourly vehicle volume, historical (2025-2026) and predicted</div>', unsafe_allow_html=True)

model = load_model()
history = load_history()

min_date = history['DateTime'].min()
max_date = history['DateTime'].max()

with st.sidebar:
    st.markdown("### Settings")
    start_ts = st.slider(
        "Forecast starting point",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=max_date.to_pydatetime(),
        format="DD/MM/YY HH:mm",
    )
    n_hours = st.slider("Hours to forecast", min_value=1, max_value=72, value=24)
    lookback_days = st.slider("History shown (days before)", min_value=1, max_value=30, value=7)

start_ts = pd.Timestamp(start_ts)

window_start = start_ts - pd.Timedelta(days=lookback_days)
hist_window = history[(history['DateTime'] > window_start) & (history['DateTime'] <= start_ts)]

forecast_df = forecast_next_hours(model, history[history['DateTime'] <= start_ts], start_ts, n_hours)

# ---------------------------------------------------------------------------
# Main chart
# ---------------------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_window['DateTime'], y=hist_window['Vehicles'],
    mode='lines', name='Historical', line=dict(color=PRIMARY, width=2.2)
))
fig.add_trace(go.Scatter(
    x=forecast_df['DateTime'], y=forecast_df['prediction'],
    mode='lines', name='Forecast', line=dict(color=ACCENT, width=2.2, dash='dash')
))
fig.add_vline(x=start_ts, line_dash="dot", line_color=SUBTEXT,
              annotation_text="Now", annotation_font_color=SUBTEXT, annotation_position="top")
fig.update_layout(
    height=440,
    plot_bgcolor=PANEL,
    paper_bgcolor=PANEL,
    font=dict(color=TEXT, family="Helvetica, Arial, sans-serif"),
    xaxis=dict(title="Date / Time", gridcolor=BORDER, zeroline=False),
    yaxis=dict(title="Vehicles / hour", gridcolor=BORDER, zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Key figures</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

current_value = f"{hist_window['Vehicles'].iloc[-1]:.0f}" if len(hist_window) else "N/A"
peak_row = forecast_df.loc[forecast_df['prediction'].idxmax()]

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current traffic</div>
        <div class="metric-value">{current_value}</div>
        <div class="metric-sub">vehicles / hour</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Peak forecast</div>
        <div class="metric-value">{peak_row['prediction']:.0f}</div>
        <div class="metric-sub">at {peak_row['DateTime'].strftime('%b %d, %H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average forecast</div>
        <div class="metric-value">{forecast_df['prediction'].mean():.0f}</div>
        <div class="metric-sub">vehicles / hour</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Alert banner
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Outlook</div>', unsafe_allow_html=True)
threshold = history['Vehicles'].quantile(0.85)
high_traffic_hours = forecast_df[forecast_df['prediction'] > threshold]

if len(high_traffic_hours) > 0:
    hours_list = ", ".join(high_traffic_hours['DateTime'].dt.strftime('%d/%m %H:%M'))
    st.markdown(f"""
    <div class="status-banner status-alert">
        Elevated traffic expected (above {threshold:.0f} veh/h) at: {hours_list}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-banner status-clear">
        No unusual traffic peaks expected over this period.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Detail table
# ---------------------------------------------------------------------------
with st.expander("View forecast detail"):
    st.dataframe(
        forecast_df.assign(prediction=forecast_df['prediction'].round(1)),
        use_container_width=True
    )
