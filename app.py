import io
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def fit_power_law(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if np.any(xs <= 0) or np.any(ys <= 0):
        raise ValueError("Power-law fit requires positive x and y values.")

    logx = np.log10(xs)
    logy = np.log10(ys)
    b, a = np.polyfit(logx, logy, 1)
    return float(a), float(b)


def predict_power_law(x: float, a: float, b: float) -> float:
    return float(10 ** (a + b * math.log10(float(x))))


def load_aircraft(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["mtow_lbs"] = pd.to_numeric(df["mtow_lbs"], errors="coerce")
    return df


def load_calibration(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["mtow_lbs"] = pd.to_numeric(df["mtow_lbs"], errors="coerce")
    df["total_nre_usd_m"] = pd.to_numeric(df["total_nre_usd_m"], errors="coerce")
    df = df.dropna(subset=["mtow_lbs", "total_nre_usd_m"])
    return df


def build_export_table(summary: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": str(k), "value": "" if v is None else str(v)} for k, v in summary.items()],
        columns=["metric", "value"],
    )


st.set_page_config(page_title="Tamarack STC Cost & Schedule Estimator", layout="wide")
st.title("Tamarack STC Cost & Schedule Estimator")

with st.sidebar:
    st.header("Inputs")

    aircraft_df = load_aircraft("data/aircraft.csv")
    aircraft_names = ["(Custom)"] + aircraft_df["aircraft"].fillna("").tolist()
    selected_aircraft = st.selectbox("Aircraft", options=aircraft_names, index=0)

    if selected_aircraft != "(Custom)":
        mtow_default = float(
            aircraft_df.loc[aircraft_df["aircraft"] == selected_aircraft, "mtow_lbs"].iloc[0]
        )
    else:
        mtow_default = 10400.0

    mtow_lbs = st.number_input(
        "MTOW (lbs)", min_value=1.0, value=float(mtow_default), step=100.0, format="%.0f"
    )

    st.subheader("Aircraft acquisition")
    acquisition_mode = st.selectbox("Mode", options=["Lease", "Purchase"], index=0)

    acquisition_months = st.number_input(
        "Months held (including STC program)", min_value=0, value=18, step=1
    )

    if acquisition_mode == "Lease":
        lease_usd_per_month = st.number_input(
            "Lease cost (USD/month)", min_value=0.0, value=75000.0, step=5000.0
        )
        acquisition_cost_usd = float(lease_usd_per_month) * float(acquisition_months)
        resale_value_usd = 0.0
    else:
        purchase_price_usd = st.number_input(
            "Purchase price (USD)", min_value=0.0, value=6000000.0, step=250000.0
        )
        resale_fraction = st.slider("Resale fraction", min_value=0.0, max_value=1.0, value=0.9)
        resale_value_usd = float(purchase_price_usd) * float(resale_fraction)
        acquisition_cost_usd = float(purchase_price_usd) - float(resale_value_usd)

    st.subheader("Readiness & inventory")
    tooling_usd = st.number_input("Tooling (USD)", min_value=0.0, value=750000.0, step=50000.0)
    production_readiness_usd = st.number_input(
        "Production readiness (USD)", min_value=0.0, value=500000.0, step=50000.0
    )
    inventory_usd = st.number_input(
        "Required inventory to start installs (USD)",
        min_value=0.0,
        value=1500000.0,
        step=50000.0,
    )

    st.subheader("Schedule")
    schedule_months = st.number_input(
        "Estimated schedule (months)", min_value=0.0, value=18.0, step=1.0
    )

cal_df = load_calibration("data/nre_calibration.csv")
a, b = fit_power_law(cal_df["mtow_lbs"].to_numpy(), cal_df["total_nre_usd_m"].to_numpy())

base_nre_usd_m = predict_power_law(float(mtow_lbs), a, b)
base_nre_usd = base_nre_usd_m * 1_000_000.0

all_in_cost_usd = base_nre_usd + float(acquisition_cost_usd) + float(tooling_usd) + float(
    production_readiness_usd
) + float(inventory_usd)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Results")

    summary = {
        "Aircraft": selected_aircraft,
        "MTOW (lbs)": float(mtow_lbs),
        "Base certification NRE (USD M)": float(base_nre_usd_m),
        "Acquisition net cost (USD)": float(acquisition_cost_usd),
        "Tooling (USD)": float(tooling_usd),
        "Production readiness (USD)": float(production_readiness_usd),
        "Inventory (USD)": float(inventory_usd),
        "All-in cost to readiness (USD)": float(all_in_cost_usd),
        "Estimated schedule (months)": float(schedule_months),
    }

    st.dataframe(build_export_table(summary), use_container_width=True, hide_index=True)

    export_df = pd.DataFrame([summary])
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="stc_estimate.csv")

    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Estimate")
        cal_df.to_excel(writer, index=False, sheet_name="NRE_Calibration")
    st.download_button(
        "Download Excel",
        data=xlsx_buffer.getvalue(),
        file_name="stc_estimate.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with col2:
    st.subheader("MTOW vs Total NRE (Power-law fit)")

    x_grid = np.linspace(cal_df["mtow_lbs"].min() * 0.8, cal_df["mtow_lbs"].max() * 1.1, 200)
    y_grid = np.array([predict_power_law(float(x), a, b) for x in x_grid])

    fit_df = pd.DataFrame({"mtow_lbs": x_grid, "total_nre_usd_m": y_grid, "series": "fit"})
    pts_df = cal_df.assign(series="calibration")
    cur_df = pd.DataFrame(
        {
            "mtow_lbs": [float(mtow_lbs)],
            "total_nre_usd_m": [float(base_nre_usd_m)],
            "series": ["selected"],
        }
    )

    plot_df = pd.concat([fit_df, pts_df, cur_df], ignore_index=True)

    line = (
        alt.Chart(plot_df[plot_df["series"] == "fit"])
        .mark_line()
        .encode(x=alt.X("mtow_lbs:Q", title="MTOW (lbs)"), y=alt.Y("total_nre_usd_m:Q", title="Total NRE (USD M)"))
    )

    points = (
        alt.Chart(plot_df[plot_df["series"] != "fit"])
        .mark_point(filled=True, size=120)
        .encode(
            x="mtow_lbs:Q",
            y="total_nre_usd_m:Q",
            color=alt.Color("series:N", legend=alt.Legend(title="")),
            tooltip=["series:N", "mtow_lbs:Q", "total_nre_usd_m:Q"],
        )
    )

    st.altair_chart((line + points).interactive(), use_container_width=True)

st.caption("Calibration points and aircraft list can be edited in the data/ folder.")
