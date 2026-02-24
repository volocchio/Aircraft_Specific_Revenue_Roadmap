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
    if "fuel_burn_gph" in df.columns:
        df["fuel_burn_gph"] = pd.to_numeric(df["fuel_burn_gph"], errors="coerce")
    if "fleet_size" in df.columns:
        df["fleet_size"] = pd.to_numeric(df["fleet_size"], errors="coerce").fillna(0).astype(int)
    if "hull_value_usd" in df.columns:
        df["hull_value_usd"] = pd.to_numeric(df["hull_value_usd"], errors="coerce")
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


def fmt_value(key: str, val) -> str:
    if val is None:
        return ""
    k = str(key).lower()
    # plain text fields
    if "aircraft" in k and "market" not in k:
        return str(val)
    # unit-less counts / ratios
    if "fleet" in k:
        return f"{int(val):,}"
    if "multiple" in k or "ratio" in k:
        return f"{float(val):.2f}x"
    if "hours" in k:
        return f"{float(val):,.0f} hrs"
    if "months" in k:
        return f"{float(val):.0f} mo"
    # flight test dollar items — keep as whole dollars
    if "flight test" in k and ("cost" in k or "fuel" in k or "crew" in k or "total" in k):
        return f"${float(val):,.0f}"
    # all other dollar amounts → always $M
    if "usd" in k or "cost" in k or "price" in k or "revenue" in k or "market" in k or "value" in k or "nre" in k or "readiness" in k or "tooling" in k or "inventory" in k or "acquisition" in k:
        return f"${float(val)/1_000_000:,.1f}M"
    # fallback
    try:
        return f"{float(val):,g}"
    except (ValueError, TypeError):
        return str(val)


def build_display_table(summary: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Metric": k, "Value": fmt_value(k, v)} for k, v in summary.items()],
        columns=["Metric", "Value"],
    )


st.set_page_config(page_title="Tamarack STC Cost & Schedule Estimator", layout="wide")
st.title("Tamarack STC Cost & Schedule Estimator")

with st.sidebar:
    st.header("Inputs")

    aircraft_df = load_aircraft("data/aircraft.csv")

    all_categories = sorted(aircraft_df["notes"].dropna().unique().tolist())
    category_options = ["All"] + all_categories
    selected_category = st.selectbox("Category", options=category_options, index=0)

    if selected_category == "All":
        filtered_df = aircraft_df
    else:
        filtered_df = aircraft_df[aircraft_df["notes"] == selected_category]

    aircraft_names = ["(Custom)"] + filtered_df["aircraft"].fillna("").tolist()
    selected_aircraft = st.selectbox("Aircraft", options=aircraft_names, index=0)

    if selected_aircraft != "(Custom)":
        _ac_row = aircraft_df.loc[aircraft_df["aircraft"] == selected_aircraft]
        mtow_default = float(_ac_row["mtow_lbs"].iloc[0])
        fleet_size_default = int(_ac_row["fleet_size"].iloc[0]) if "fleet_size" in _ac_row.columns else 0
        hull_value_default = float(_ac_row["hull_value_usd"].iloc[0]) if "hull_value_usd" in _ac_row.columns and not _ac_row["hull_value_usd"].isna().all() else 5000000.0
    else:
        mtow_default = 10400.0
        fleet_size_default = 0
        hull_value_default = 5000000.0

    mtow_lbs = st.number_input(
        "MTOW (lbs)", min_value=1.0, value=float(mtow_default), step=100.0, format="%.0f"
    )

    st.subheader("Aircraft acquisition")
    acquisition_mode = st.selectbox("Mode", options=["Lease", "Purchase"], index=0)

    acquisition_months = st.number_input(
        "Months held (including STC program)", min_value=0, value=24, step=1
    )

    if acquisition_mode == "Lease":
        hull_value_usd = st.number_input(
            "Hull value (USD)", min_value=0.0, value=hull_value_default, step=250000.0, format="%.0f"
        )
        lease_rate_pct = st.slider(
            "Lease rate (% of hull/month)",
            min_value=0.5, max_value=3.0, value=1.0, step=0.1, format="%.1f%%"
        )
        lease_usd_per_month = hull_value_usd * (lease_rate_pct / 100.0)
        acquisition_cost_usd = lease_usd_per_month * float(acquisition_months)
        resale_value_usd = 0.0
        st.caption(
            f"Monthly lease: **${lease_usd_per_month:,.0f}**/mo  ·  "
            f"Total acquisition cost: **${acquisition_cost_usd/1e6:,.1f}M**"
        )
    else:
        purchase_price_usd = st.number_input(
            "Purchase price (USD)", min_value=0.0, value=hull_value_default, step=250000.0, format="%.0f"
        )
        resale_fraction = st.slider("Resale fraction", min_value=0.0, max_value=1.0, value=0.9, format="%.2f")
        resale_value_usd = float(purchase_price_usd) * float(resale_fraction)
        acquisition_cost_usd = float(purchase_price_usd) - float(resale_value_usd)
        st.caption(
            f"Resale value: **${resale_value_usd/1e6:,.1f}M**  ·  "
            f"Net acquisition cost: **${acquisition_cost_usd/1e6:,.1f}M**"
        )

    st.subheader("Readiness & inventory")
    tooling_usd = st.number_input("Tooling (USD)", min_value=0.0, value=750000.0, step=50000.0, format="%.0f")
    inventory_usd = st.number_input(
        "Required inventory to start installs (USD)",
        min_value=0.0,
        value=1500000.0,
        step=50000.0,
        format="%.0f",
    )
    prod_readiness_pct = st.slider(
        "Production readiness (% of cert NRE)",
        min_value=0,
        max_value=60,
        value=25,
        step=1,
        format="%d%%",
    )

    st.subheader("Flight testing")

    if selected_aircraft != "(Custom)" and "fuel_burn_gph" in aircraft_df.columns:
        _fb_row = aircraft_df.loc[aircraft_df["aircraft"] == selected_aircraft, "fuel_burn_gph"]
        fuel_burn_default = float(_fb_row.iloc[0]) if not _fb_row.empty else 150.0
    else:
        fuel_burn_default = 150.0

    baseline_hrs = st.number_input(
        "Baseline flight test hours", min_value=0.0, value=50.0, step=5.0, format="%.0f"
    )
    cert_hrs = st.number_input(
        "Certification flight test hours", min_value=0.0, value=100.0, step=5.0, format="%.0f"
    )
    total_flight_test_hrs = baseline_hrs + cert_hrs

    fuel_burn_gph = st.number_input(
        "Fuel burn (gal/hr)", min_value=0.0, value=fuel_burn_default, step=10.0, format="%.0f"
    )
    fuel_price_per_gal = st.number_input(
        "Fuel price (USD/gal)", min_value=0.0, value=6.50, step=0.25, format="%.2f"
    )
    crew_cost_per_hr = st.number_input(
        "Crew cost (USD/hr)", min_value=0.0, value=1500.0, step=100.0, format="%.0f"
    )

    flight_test_fuel_usd = total_flight_test_hrs * fuel_burn_gph * fuel_price_per_gal
    flight_test_crew_usd = total_flight_test_hrs * crew_cost_per_hr
    flight_test_total_usd = flight_test_fuel_usd + flight_test_crew_usd

    st.caption(
        f"Est. flight test cost: **${flight_test_total_usd:,.0f}** "
        f"({total_flight_test_hrs:.0f} hrs total)"
    )

    st.subheader("Market")
    kit_price_usd = st.number_input(
        "Price to customer (USD)", min_value=0.0, value=500000.0, step=25000.0, format="%.0f"
    )
    fleet_size = st.number_input(
        "Fleet size (global in-service)",
        min_value=0,
        value=fleet_size_default,
        step=10,
    )
    market_penetration_pct = st.slider(
        "Expected market penetration",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        format="%d%%",
    )

    st.subheader("Schedule")
    schedule_months = st.number_input(
        "Estimated schedule (months)", min_value=1.0, value=24.0, step=1.0, format="%.0f"
    )
    _current_year = pd.Timestamp.now().year
    program_start_year = st.number_input(
        "Program start year", min_value=2020, max_value=2040, value=_current_year, step=1, format="%d"
    )
    program_start_quarter = st.selectbox(
        "Program start quarter", options=["Q1", "Q2", "Q3", "Q4"], index=0
    )
    _q_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[program_start_quarter]
    program_start_date = pd.Timestamp(year=int(program_start_year), month=_q_month, day=1)
    program_end_date = program_start_date + pd.DateOffset(months=int(schedule_months))
    st.caption(
        f"Program: **{program_start_date.strftime('%b %Y')}** → **{program_end_date.strftime('%b %Y')}** "
        f"({int(schedule_months)} mo)"
    )

cal_df = load_calibration("data/nre_calibration.csv")
a, b = fit_power_law(cal_df["mtow_lbs"].to_numpy(), cal_df["total_nre_usd_m"].to_numpy())

base_nre_usd_m = predict_power_law(float(mtow_lbs), a, b)
base_nre_usd = base_nre_usd_m * 1_000_000.0

production_readiness_usd = base_nre_usd * (float(prod_readiness_pct) / 100.0)

all_in_cost_usd = base_nre_usd + float(acquisition_cost_usd) + float(tooling_usd) + float(
    production_readiness_usd
) + float(inventory_usd) + float(flight_test_total_usd)

tam_usd = float(fleet_size) * float(kit_price_usd)
addressable_revenue_usd = tam_usd * (float(market_penetration_pct) / 100.0)
roi_multiple = addressable_revenue_usd / all_in_cost_usd if all_in_cost_usd > 0 else 0.0

tab_results, tab_schedule = st.tabs(["Results", "Schedule"])

col1, col2 = tab_results.columns([1.2, 1])

with col1:
    st.subheader("Market Opportunity")
    mkt_col1, mkt_col2, mkt_col3 = st.columns(3)
    mkt_col1.metric("Fleet size", f"{int(fleet_size):,}")
    mkt_col2.metric("Total Addressable Market", f"${tam_usd/1e6:,.1f}M")
    mkt_col3.metric(
        f"Revenue @ {market_penetration_pct}% penetration",
        f"${addressable_revenue_usd/1e6:,.1f}M",
    )

    roi_col1, roi_col2 = st.columns(2)
    roi_col1.metric("All-in investment", f"${all_in_cost_usd/1e6:,.1f}M")
    roi_col2.metric("Revenue / Investment", f"{roi_multiple:.1f}x")

    st.subheader("Cost Detail")

    summary = {
        "Aircraft": selected_aircraft,
        "MTOW (lbs)": float(mtow_lbs),
        "Fleet size (global)": int(fleet_size),
        "Price to customer (USD)": float(kit_price_usd),
        "Total addressable market (USD)": float(tam_usd),
        f"Addressable revenue @ {market_penetration_pct}% (USD)": float(addressable_revenue_usd),
        "Revenue / investment multiple": float(roi_multiple),
        "Base certification NRE (USD M)": float(base_nre_usd_m),
        "Acquisition net cost (USD)": float(acquisition_cost_usd),
        "Tooling (USD)": float(tooling_usd),
        "Production readiness (USD)": float(production_readiness_usd),
        "Inventory (USD)": float(inventory_usd),
        "Flight test hours (baseline)": float(baseline_hrs),
        "Flight test hours (certification)": float(cert_hrs),
        "Flight test fuel cost (USD)": float(flight_test_fuel_usd),
        "Flight test crew cost (USD)": float(flight_test_crew_usd),
        "Flight test total (USD)": float(flight_test_total_usd),
        "All-in cost to readiness (USD)": float(all_in_cost_usd),
        "Estimated schedule (months)": float(schedule_months),
    }

    st.dataframe(build_display_table(summary), use_container_width=True, hide_index=True)

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
    st.subheader("MTOW vs Total NRE (log-log)")

    x_grid = np.logspace(
        np.log10(5000.0),
        np.log10(cal_df["mtow_lbs"].max() * 1.1),
        200,
    )
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
        .encode(
            x=alt.X("mtow_lbs:Q", title="MTOW (lbs)", scale=alt.Scale(type="log")),
            y=alt.Y("total_nre_usd_m:Q", title="Total NRE (USD M)", scale=alt.Scale(type="log")),
        )
    )

    points = (
        alt.Chart(plot_df[plot_df["series"] != "fit"])
        .mark_point(filled=True, size=120)
        .encode(
            x=alt.X("mtow_lbs:Q", scale=alt.Scale(type="log")),
            y=alt.Y("total_nre_usd_m:Q", scale=alt.Scale(type="log")),
            color=alt.Color("series:N", legend=alt.Legend(title="")),
            tooltip=["series:N", "mtow_lbs:Q", "total_nre_usd_m:Q"],
        )
    )

    st.altair_chart((line + points).interactive(), use_container_width=True)

tab_results.caption("Calibration points and aircraft list can be edited in the data/ folder.")

with tab_schedule:
    st.subheader(
        f"STC Program Schedule — {program_start_date.strftime('%b %Y')} to {program_end_date.strftime('%b %Y')}"
    )

    # Strawman proportions expressed as fractions of a 666-day (22-month) baseline
    _BASELINE_DAYS = 666.0
    _scale = (float(schedule_months) * 30.44) / _BASELINE_DAYS

    def _sd(d: int) -> int:
        return max(0, round(d * _scale))

    def _dur(d: int) -> int:
        return max(1, round(d * _scale))

    _gantt_data = [
        {"Phase": "Design Effort", "Task": "Design", "Start_day": _sd(0), "Duration_days": _dur(200)},
        {"Phase": "Design Effort", "Task": "Proof of Concept / Prototype", "Start_day": _sd(30), "Duration_days": _dur(90)},
        {"Phase": "Design Effort", "Task": "Prototype Flight Test", "Start_day": _sd(120), "Duration_days": _dur(20)},
        {"Phase": "Design Effort", "Task": "MDL", "Start_day": _sd(60), "Duration_days": _dur(150)},
        {"Phase": "Design Effort", "Task": "Tooling", "Start_day": _sd(210), "Duration_days": _dur(100)},
        {"Phase": "Design Effort", "Task": "Production", "Start_day": _sd(310), "Duration_days": _dur(1)},
        {"Phase": "FAA Certification", "Task": "Cert Plans & Cert Basis", "Start_day": _sd(120), "Duration_days": _dur(30)},
        {"Phase": "FAA Certification", "Task": "1309 Compliance", "Start_day": _sd(150), "Duration_days": _dur(130)},
        {"Phase": "FAA Certification", "Task": "Prepare Test Plans", "Start_day": _sd(120), "Duration_days": _dur(120)},
        {"Phase": "FAA Certification", "Task": "Baseline Characteristics Testing", "Start_day": _sd(240), "Duration_days": _dur(45)},
        {"Phase": "FAA Certification", "Task": "Baseline Performance Testing", "Start_day": _sd(285), "Duration_days": _dur(60)},
        {"Phase": "FAA Certification", "Task": "Qualification Testing", "Start_day": _sd(210), "Duration_days": _dur(180)},
        {"Phase": "FAA Certification", "Task": "Modified GT & Reporting", "Start_day": _sd(390), "Duration_days": _dur(60)},
        {"Phase": "FAA Certification", "Task": "Flutter Analysis", "Start_day": _sd(390), "Duration_days": _dur(150)},
        {"Phase": "FAA Certification", "Task": "Envelope Expansion", "Start_day": _sd(450), "Duration_days": _dur(10)},
        {"Phase": "FAA Certification", "Task": "Loads Flight Tests & Reporting", "Start_day": _sd(460), "Duration_days": _dur(60)},
        {"Phase": "FAA Certification", "Task": "Loads Reports", "Start_day": _sd(490), "Duration_days": _dur(30)},
        {"Phase": "FAA Certification", "Task": "Modified Flight Tests & Reporting", "Start_day": _sd(460), "Duration_days": _dur(90)},
        {"Phase": "FAA Certification", "Task": "Ice Flights & Reporting", "Start_day": _sd(490), "Duration_days": _dur(90)},
        {"Phase": "FAA Certification", "Task": "Modified Performance Testing", "Start_day": _sd(520), "Duration_days": _dur(90)},
        {"Phase": "FAA Certification", "Task": "Stress Reports", "Start_day": _sd(490), "Duration_days": _dur(200)},
        {"Phase": "FAA Certification", "Task": "Damage Tolerance Reports", "Start_day": _sd(490), "Duration_days": _dur(200)},
        {"Phase": "FAA Certification", "Task": "Structural Testing", "Start_day": _sd(490), "Duration_days": _dur(130)},
        {"Phase": "FAA Certification", "Task": "STC Issuance", "Start_day": _sd(665), "Duration_days": _dur(1)},
    ]

    gantt_df = pd.DataFrame(_gantt_data)
    gantt_df["Start"] = program_start_date + pd.to_timedelta(gantt_df["Start_day"], unit="D")
    gantt_df["Finish"] = gantt_df["Start"] + pd.to_timedelta(gantt_df["Duration_days"], unit="D")
    gantt_df["Start_label"] = gantt_df["Start"].dt.strftime("%b %Y")
    gantt_df["Finish_label"] = gantt_df["Finish"].dt.strftime("%b %Y")

    gantt_chart = (
        alt.Chart(gantt_df)
        .mark_bar(height={"band": 0.7})
        .encode(
            x=alt.X("Start:T", title="Date"),
            x2=alt.X2("Finish:T"),
            y=alt.Y(
                "Task:N",
                sort=None,
                title="",
                axis=alt.Axis(labelLimit=280, labelFontSize=12),
            ),
            color=alt.Color("Phase:N", legend=alt.Legend(title="Phase")),
            tooltip=[
                alt.Tooltip("Task:N", title="Task"),
                alt.Tooltip("Phase:N", title="Phase"),
                alt.Tooltip("Start_label:N", title="Start"),
                alt.Tooltip("Finish_label:N", title="Finish"),
                alt.Tooltip("Duration_days:Q", title="Duration (days)"),
            ],
        )
        .properties(height=700)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(gantt_chart, use_container_width=True)
    st.caption(
        f"Schedule scaled to **{int(schedule_months)} months** starting **{program_start_quarter} {int(program_start_year)}**. "
        "Proportions based on A320 STC strawman."
    )
