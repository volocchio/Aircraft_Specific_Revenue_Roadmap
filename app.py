import io
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)


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


def chart_to_png(chart: alt.Chart, width: int = 600, height: int = 300) -> bytes:
    buf = io.BytesIO()
    chart.save(buf, format="png", scale_factor=1.5)
    return buf.getvalue()


def build_pdf(summary: dict, aircraft: str, program_start: str, program_end: str,
             chart_pngs: list[tuple[str, bytes]] | None = None,
             fin_tables: list[tuple[str, "pd.DataFrame"]] | None = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title", parent=styles["Heading1"], fontSize=16, spaceAfter=6
    )
    section_style = ParagraphStyle(
        "section", parent=styles["Heading2"], fontSize=12,
        spaceBefore=12, spaceAfter=4, textColor=colors.HexColor("#1f4e79")
    )
    body_style = styles["Normal"]

    tbl_style = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f7fc")]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ])

    story = []
    story.append(Paragraph("Tamarack STC Cost & Schedule Estimator", title_style))
    story.append(Paragraph(
        f"Aircraft: {aircraft}  |  Program: {program_start} → {program_end}",
        body_style
    ))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("Program Summary", section_style))
    rows = [["Metric", "Value"]]
    for k, v in summary.items():
        rows.append([str(k), fmt_value(str(k), v)])
    col_w = [4.0*inch, 2.5*inch]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(tbl_style)
    story.append(t)

    if fin_tables:
        for _ftitle, _fdf in fin_tables:
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph(_ftitle, section_style))
            _frows = [list(_fdf.columns)] + _fdf.values.tolist()
            _ft = Table(_frows, colWidths=[4.5*inch, 2.0*inch], repeatRows=1)
            _ft.setStyle(tbl_style)
            story.append(_ft)

    if chart_pngs:
        for _title, _png in chart_pngs:
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph(_title, section_style))
            _img_buf = io.BytesIO(_png)
            _rl_img = RLImage(_img_buf, width=6.5*inch, height=3.25*inch)
            story.append(_rl_img)

    doc.build(story)
    return buf.getvalue()


def fmt_value(key: str, val) -> str:
    if val is None:
        return ""
    k = str(key).lower()
    # plain text fields — only the aircraft name row, not cost rows
    if "aircraft" in k and "market" not in k and "usd" not in k and "cost" not in k and "acquisition" not in k:
        return str(val)
    # unit-less counts / ratios
    if "fleet" in k:
        return f"{int(val):,}"
    if "multiple" in k or "ratio" in k or "/ investment" in k:
        return f"{float(val):.2f}x"
    if "hours" in k:
        return f"{float(val):,.0f} hrs"
    if "months" in k:
        return f"{float(val):.0f} mo"
    # unit counts
    if k.startswith("units sold"):
        return f"{int(val):,}"
    # flight test dollar items — keep as whole dollars
    if "flight test" in k and ("cost" in k or "fuel" in k or "crew" in k or "total" in k):
        return f"${float(val):,.0f}"
    # all other dollar amounts → always $M
    if any(t in k for t in ("usd", "cost", "price", "revenue", "market", "value", "nre",
                             "readiness", "tooling", "inventory", "acquisition",
                             "royalt", "license", "cogs", "investment")):
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
    st.caption(f"Program: {program_start_date.strftime('%b %Y')} → {program_end_date.strftime('%b %Y')} ({int(schedule_months)} mo)")

    st.subheader("Aircraft acquisition")
    acquisition_mode = st.selectbox("Mode", options=["Lease", "Purchase"], index=0)
    st.caption("Aircraft held during STC program only — released at certification.")

    if acquisition_mode == "Lease":
        hull_value_usd = st.number_input(
            "Hull value ($M)", min_value=0.0, value=round(hull_value_default/1e6, 2), step=0.25, format="%.2f"
        ) * 1e6
        lease_rate_pct = st.slider(
            "Lease rate (% of hull/month)",
            min_value=0.5, max_value=3.0, value=1.0, step=0.1, format="%.1f%%"
        )
        lease_usd_per_month = hull_value_usd * (lease_rate_pct / 100.0)
        # Lease runs for schedule_months only — ends at STC issuance
        acquisition_cost_usd = lease_usd_per_month * schedule_months
        resale_value_usd = 0.0
        st.caption(f"\\${lease_usd_per_month:,.0f}/mo × {int(schedule_months)} mo = \\${acquisition_cost_usd/1e6:,.1f}M total lease cost")
    else:
        purchase_price_usd = st.number_input(
            "Purchase price ($M)", min_value=0.0, value=round(hull_value_default/1e6, 2), step=0.25, format="%.2f"
        ) * 1e6
        resale_fraction = st.slider("Resale value at STC (% of purchase)", min_value=0.0, max_value=1.0, value=0.9, format="%.2f")
        resale_value_usd = float(purchase_price_usd) * float(resale_fraction)
        acquisition_cost_usd = float(purchase_price_usd) - float(resale_value_usd)
        st.caption(f"Purchase \\${purchase_price_usd/1e6:,.1f}M − resale \\${resale_value_usd/1e6:,.1f}M = net cost \\${acquisition_cost_usd/1e6:,.1f}M")

    st.subheader("Engineering")
    eng_rate_usd_per_hr = st.slider(
        "Engineering rate (USD/hr)",
        min_value=100, max_value=350, value=175, step=5, format="$%d"
    )

    st.subheader("Readiness & inventory")
    tooling_usd = st.number_input("Tooling ($M)", min_value=0.0, value=0.75, step=0.05, format="%.2f") * 1e6
    inventory_usd = st.number_input(
        "Required inventory to start installs ($M)",
        min_value=0.0,
        value=1.5,
        step=0.1,
        format="%.2f",
    ) * 1e6
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

    st.caption(f"Est. flight test cost: \\${flight_test_total_usd:,.0f} ({total_flight_test_hrs:.0f} hrs total)")

    st.subheader("Market")
    kit_price_usd = st.number_input(
        "Sales price per unit ($M)", min_value=0.0, value=0.5, step=0.05, format="%.2f"
    ) * 1e6
    cogs_per_unit_usd = st.number_input(
        "Est. COGS per unit ($M)", min_value=0.0, value=0.15, step=0.01, format="%.2f"
    ) * 1e6
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
    license_fee_usd = st.number_input(
        "License fee ($M, one-time)", min_value=0.0, value=5.0, step=0.5, format="%.1f"
    ) * 1e6
    royalty_pct = st.slider(
        "Royalty rate (% of gross revenue)",
        min_value=0, max_value=30, value=10, step=1, format="%d%%"
    )
    _tam = fleet_size * kit_price_usd
    _gross = fleet_size * (market_penetration_pct / 100.0) * kit_price_usd
    st.caption(f"TAM: \\${_tam/1e6:,.1f}M  ·  Gross revenue @ {market_penetration_pct}%: \\${_gross/1e6:,.1f}M")


cal_df = load_calibration("data/nre_calibration.csv")
# Scale calibration NRE values by user engineering rate vs baseline $175/hr
_rate_scale = float(eng_rate_usd_per_hr) / 175.0
_scaled_nre = cal_df["total_nre_usd_m"].to_numpy() * _rate_scale
a, b = fit_power_law(cal_df["mtow_lbs"].to_numpy(), _scaled_nre)

base_nre_usd_m = predict_power_law(float(mtow_lbs), a, b)
base_nre_usd = base_nre_usd_m * 1_000_000.0

production_readiness_usd = base_nre_usd * (float(prod_readiness_pct) / 100.0)
# Surface readiness value back into sidebar as a caption
with st.sidebar:
    st.caption(f"Production readiness: \\${production_readiness_usd/1e6:,.1f}M ({prod_readiness_pct}% of \\${base_nre_usd/1e6:,.1f}M NRE)")

all_in_cost_usd = base_nre_usd + float(acquisition_cost_usd) + float(tooling_usd) + float(
    production_readiness_usd
) + float(inventory_usd) + float(flight_test_total_usd)

units_sold = float(fleet_size) * (float(market_penetration_pct) / 100.0)
tam_usd = float(fleet_size) * float(kit_price_usd)
gross_revenue_usd = units_sold * float(kit_price_usd)
cogs_total_usd = units_sold * float(cogs_per_unit_usd)
royalties_usd = gross_revenue_usd * (float(royalty_pct) / 100.0)
net_revenue_usd = gross_revenue_usd - cogs_total_usd - royalties_usd - float(license_fee_usd)
roi_multiple = net_revenue_usd / all_in_cost_usd if all_in_cost_usd > 0 else 0.0
addressable_revenue_usd = gross_revenue_usd

# ── Financial statement data (built here so PDF can access before tabs render) ──
_rev_ramp_years = 3
_fin_rev_periods = pd.date_range(
    start=program_end_date, periods=_rev_ramp_years * 4 + 1, freq="QS"
)
_fin_total_rev_q = len(_fin_rev_periods) - 1
_fin_ramp_weights = [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.125, 0.125, 0.125, 0.125]
_fin_ramp_weights = (_fin_ramp_weights + [1/_fin_total_rev_q] * max(0, _fin_total_rev_q - len(_fin_ramp_weights)))[:_fin_total_rev_q]
_fin_wsum = sum(_fin_ramp_weights)
_fin_ramp_weights = [w / _fin_wsum for w in _fin_ramp_weights]

_gross_margin    = gross_revenue_usd - cogs_total_usd
_ebit            = _gross_margin - royalties_usd - license_fee_usd - all_in_cost_usd
_cf_ops          = gross_revenue_usd - cogs_total_usd - royalties_usd - license_fee_usd
_cf_invest       = -(base_nre_usd + tooling_usd + float(flight_test_total_usd) +
                     acquisition_cost_usd + inventory_usd + production_readiness_usd)
_cf_net          = _cf_ops + _cf_invest

_is_rows = [
    ("Gross Revenue",            f"${gross_revenue_usd/1e6:,.1f}M"),
    ("Cost of Goods Sold",       f"(${cogs_total_usd/1e6:,.1f}M)"),
    ("Gross Margin",             f"${_gross_margin/1e6:,.1f}M"),
    ("Royalties",                f"(${royalties_usd/1e6:,.1f}M)"),
    ("License Fee",              f"(${license_fee_usd/1e6:,.1f}M)"),
    ("Program Investment",       f"(${all_in_cost_usd/1e6:,.1f}M)"),
    ("Net Income",               f"${_ebit/1e6:,.1f}M"),
]
_is_df = pd.DataFrame(_is_rows, columns=["Line Item", "Amount ($M)"])

_cf_stmt_rows = [
    ("Operating Activities", ""),
    ("  Revenue collected",        f"${gross_revenue_usd/1e6:,.1f}M"),
    ("  COGS paid",                f"(${cogs_total_usd/1e6:,.1f}M)"),
    ("  Royalties paid",           f"(${royalties_usd/1e6:,.1f}M)"),
    ("  License fee paid",         f"(${license_fee_usd/1e6:,.1f}M)"),
    ("Net cash from operations",   f"${_cf_ops/1e6:,.1f}M"),
    ("", ""),
    ("Investing / Development", ""),
    ("  Certification NRE",        f"(${base_nre_usd/1e6:,.1f}M)"),
    ("  Production readiness",     f"(${production_readiness_usd/1e6:,.1f}M)"),
    ("  Tooling",                  f"(${tooling_usd/1e6:,.1f}M)"),
    ("  Flight testing",           f"(${float(flight_test_total_usd)/1e6:,.1f}M)"),
    ("  Aircraft acquisition",     f"(${acquisition_cost_usd/1e6:,.1f}M)"),
    ("  Inventory",                f"(${inventory_usd/1e6:,.1f}M)"),
    ("Net cash from investing",    f"${_cf_invest/1e6:,.1f}M"),
    ("", ""),
    ("Net cash flow (lifetime)",   f"${_cf_net/1e6:,.1f}M"),
]
_cf_stmt_df = pd.DataFrame(_cf_stmt_rows, columns=["Line Item", "Amount ($M)"])

_assets_cash  = max(0.0, _cf_net)
_assets_inv   = inventory_usd
_assets_total = _assets_cash + _assets_inv
_liab_def     = 0.0
_equity       = _assets_total - _liab_def
_bs_rows = [
    ("ASSETS", ""),
    ("  Cash & equivalents",  f"${_assets_cash/1e6:,.1f}M"),
    ("  Inventory",           f"${_assets_inv/1e6:,.1f}M"),
    ("Total Assets",          f"${_assets_total/1e6:,.1f}M"),
    ("", ""),
    ("LIABILITIES", ""),
    ("  Deferred obligations",f"${_liab_def/1e6:,.1f}M"),
    ("Total Liabilities",     f"${_liab_def/1e6:,.1f}M"),
    ("", ""),
    ("EQUITY", ""),
    ("Retained earnings",     f"${_equity/1e6:,.1f}M"),
]
_bs_df = pd.DataFrame(_bs_rows, columns=["Line Item", "Amount ($M)"])

# Build CF time-series
_prog_quarters_fin = pd.date_range(start=program_start_date, end=program_end_date, freq="QS")
if len(_prog_quarters_fin) == 0:
    _prog_quarters_fin = pd.date_range(start=program_start_date, periods=2, freq="QS")
_spend_per_q_fin = all_in_cost_usd / len(_prog_quarters_fin)
_cf_ts_rows = []
_cum_fin = 0.0
for _q in _prog_quarters_fin:
    _cum_fin -= _spend_per_q_fin
    _cf_ts_rows.append({"Quarter": _q, "Cumulative Cash Flow ($M)": _cum_fin / 1e6, "Phase": "Investment"})
_cum_fin += float(resale_value_usd)
for _i, _q in enumerate(_fin_rev_periods[1:]):
    _q_rev = gross_revenue_usd * _fin_ramp_weights[_i]
    _q_cogs = cogs_total_usd * _fin_ramp_weights[_i]
    _q_roy = royalties_usd * _fin_ramp_weights[_i]
    _q_lic = license_fee_usd if _i == 0 else 0.0
    _cum_fin += (_q_rev - _q_cogs - _q_roy - _q_lic)
    _cf_ts_rows.append({"Quarter": _q, "Cumulative Cash Flow ($M)": _cum_fin / 1e6, "Phase": "Revenue"})
_cf_ts_df = pd.DataFrame(_cf_ts_rows)
_cf_ts_df["Quarter_label"] = _cf_ts_df["Quarter"].dt.strftime("%b %Y")
_zero_line_cf = pd.DataFrame({"Quarter": [_cf_ts_df["Quarter"].min(), _cf_ts_df["Quarter"].max()], "y": [0, 0]})
_cf_color_scale = alt.Scale(domain=["Investment", "Revenue"], range=["#d62728", "#2ca02c"])
_cf_area = (
    alt.Chart(_cf_ts_df).mark_area(opacity=0.3)
    .encode(
        x=alt.X("Quarter:T", title="Quarter",
                axis=alt.Axis(tickCount={"interval": "month", "step": 3}, format="%b %Y", labelAngle=-45)),
        y=alt.Y("Cumulative Cash Flow ($M):Q", title="Cumulative Cash Flow ($M)"),
        color=alt.Color("Phase:N", scale=_cf_color_scale, legend=alt.Legend(title="")),
    )
)
_cf_line = (
    alt.Chart(_cf_ts_df).mark_line(point=True)
    .encode(
        x=alt.X("Quarter:T",
                axis=alt.Axis(tickCount={"interval": "month", "step": 3}, format="%b %Y", labelAngle=-45)),
        y=alt.Y("Cumulative Cash Flow ($M):Q"),
        color=alt.Color("Phase:N", scale=_cf_color_scale),
        tooltip=[
            alt.Tooltip("Quarter_label:N", title="Quarter"),
            alt.Tooltip("Cumulative Cash Flow ($M):Q", title="Cum. CF ($M)", format=".1f"),
            alt.Tooltip("Phase:N", title="Phase"),
        ],
    )
)
_cf_zero = alt.Chart(_zero_line_cf).mark_rule(color="gray", strokeDash=[4, 4]).encode(y=alt.Y("y:Q"))
_cf_chart = (_cf_area + _cf_line + _cf_zero).properties(width=600, height=300)

# Build NRE chart here so it's available for PDF generation (col1 runs before col2)
_x_grid = np.logspace(np.log10(5000.0), np.log10(cal_df["mtow_lbs"].max() * 1.1), 200)
_y_grid = np.array([predict_power_law(float(x), a, b) for x in _x_grid])
_fit_df  = pd.DataFrame({"mtow_lbs": _x_grid, "total_nre_usd_m": _y_grid, "series": "fit"})
_pts_df  = cal_df.assign(series="calibration")
_cur_df  = pd.DataFrame({"mtow_lbs": [float(mtow_lbs)], "total_nre_usd_m": [float(base_nre_usd_m)], "series": ["selected"]})
_plot_df = pd.concat([_fit_df, _pts_df, _cur_df], ignore_index=True)
_nre_line = (
    alt.Chart(_plot_df[_plot_df["series"] == "fit"]).mark_line()
    .encode(
        x=alt.X("mtow_lbs:Q", title="MTOW (lbs)", scale=alt.Scale(type="log")),
        y=alt.Y("total_nre_usd_m:Q", title="Total NRE (USD M)", scale=alt.Scale(type="log")),
    )
)
_nre_points = (
    alt.Chart(_plot_df[_plot_df["series"] != "fit"]).mark_point(filled=True, size=120)
    .encode(
        x=alt.X("mtow_lbs:Q", scale=alt.Scale(type="log")),
        y=alt.Y("total_nre_usd_m:Q", scale=alt.Scale(type="log")),
        color=alt.Color("series:N", legend=alt.Legend(title="")),
        tooltip=["series:N", "mtow_lbs:Q", "total_nre_usd_m:Q"],
    )
)
_nre_chart = (_nre_line + _nre_points).properties(width=500, height=300)

tab_results, tab_finance, tab_sensitivity, tab_schedule = st.tabs(["Results", "Financial Statements", "Sensitivity", "Schedule"])

col1, col2 = tab_results.columns([1.2, 1])

with col1:
    st.subheader("Market Opportunity")
    mkt_col1, mkt_col2, mkt_col3 = st.columns(3)
    mkt_col1.metric("Fleet size", f"{int(fleet_size):,}")
    mkt_col2.metric("Total Addressable Market", f"${tam_usd/1e6:,.1f}M")
    mkt_col3.metric(f"Units sold @ {market_penetration_pct}%", f"{int(units_sold):,}")

    st.subheader("Program Investment")
    inv_col1, inv_col2, inv_col3 = st.columns(3)
    inv_col1.metric("Certification NRE", f"${base_nre_usd/1e6:,.1f}M")
    inv_col2.metric("All-in program cost", f"${all_in_cost_usd/1e6:,.1f}M")
    inv_col3.metric("Production readiness", f"${production_readiness_usd/1e6:,.1f}M")

    st.subheader("Revenue & Returns")
    pl_col1, pl_col2, pl_col3 = st.columns(3)
    pl_col1.metric("Gross revenue", f"${gross_revenue_usd/1e6:,.1f}M")
    pl_col2.metric("Net revenue", f"${net_revenue_usd/1e6:,.1f}M")
    pl_col3.metric("Net revenue / Investment", f"{roi_multiple:.1f}x")

    st.subheader("Break-Even")
    _margin_per_unit = float(kit_price_usd) - float(cogs_per_unit_usd) - float(kit_price_usd) * (float(royalty_pct) / 100.0)
    _be_units = (all_in_cost_usd + float(license_fee_usd)) / _margin_per_unit if _margin_per_unit > 0 else float("inf")
    _be_pct = (_be_units / float(fleet_size) * 100.0) if float(fleet_size) > 0 else float("inf")
    be_col1, be_col2, be_col3 = st.columns(3)
    be_col1.metric("Break-even units", f"{_be_units:,.0f}" if _be_units < 1e6 else "N/A")
    be_col2.metric("Break-even penetration", f"{_be_pct:.1f}%" if _be_pct < 200 else "N/A")
    be_col3.metric("Margin per unit", f"${_margin_per_unit/1e6:,.2f}M")

    st.subheader("Program Summary")

    summary = {
        "Aircraft": selected_aircraft,
        "MTOW (lbs)": float(mtow_lbs),
        "Fleet size (global)": int(fleet_size),
        "Sales price per unit (USD)": float(kit_price_usd),
        "Est. COGS per unit (USD)": float(cogs_per_unit_usd),
        "Total addressable market (USD)": float(tam_usd),
        f"Units sold @ {market_penetration_pct}% penetration": int(units_sold),
        "Gross revenue (USD)": float(gross_revenue_usd),
        "COGS total (USD)": float(cogs_total_usd),
        f"Royalties @ {royalty_pct}% (USD)": float(royalties_usd),
        "License fee (USD)": float(license_fee_usd),
        "Net revenue (USD)": float(net_revenue_usd),
        "Net revenue / investment": float(roi_multiple),
        "Program investment (USD)": float(all_in_cost_usd),
        "Certification NRE (USD)": float(base_nre_usd),
        "Aircraft acquisition cost (USD)": float(acquisition_cost_usd),
        "Tooling (USD)": float(tooling_usd),
        "Production readiness (USD)": float(production_readiness_usd),
        "Inventory (USD)": float(inventory_usd),
        "Flight test hours (baseline)": float(baseline_hrs),
        "Flight test hours (certification)": float(cert_hrs),
        "Flight test fuel cost (USD)": float(flight_test_fuel_usd),
        "Flight test crew cost (USD)": float(flight_test_crew_usd),
        "Flight test total (USD)": float(flight_test_total_usd),
        "Estimated schedule (months)": float(schedule_months),
    }

    st.dataframe(build_display_table(summary), use_container_width=True, hide_index=True)

    try:
        _nre_png = chart_to_png(_nre_chart)
        _cf_png  = chart_to_png(_cf_chart)
        _pdf_chart_pngs = [
            ("MTOW vs Total NRE", _nre_png),
            ("Cumulative Cash Flow", _cf_png),
        ]
    except Exception:
        _pdf_chart_pngs = None
    _pdf_bytes = build_pdf(
        summary,
        aircraft=selected_aircraft,
        program_start=program_start_date.strftime("%b %Y"),
        program_end=program_end_date.strftime("%b %Y"),
        chart_pngs=_pdf_chart_pngs,
        fin_tables=[
            ("Income Statement (Program Lifetime)", _is_df),
            ("Cash Flow Statement (Program Lifetime)", _cf_stmt_df),
            ("Balance Sheet (at Program Completion)", _bs_df),
        ],
    )
    st.download_button(
        "Download PDF",
        data=_pdf_bytes,
        file_name="stc_estimate.pdf",
        mime="application/pdf",
    )

with col2:
    st.subheader("MTOW vs Total NRE (log-log)")
    st.altair_chart(_nre_chart.interactive(), use_container_width=True)

tab_results.caption("Calibration points and aircraft list can be edited in the data/ folder.")

with tab_finance:
    st.subheader(f"Financial Statements — {program_start_quarter} {int(program_start_year)}")
    st.markdown("### Income Statement (Program Lifetime, $M)")
    st.dataframe(_is_df, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("### Cash Flow Statement (Program Lifetime, $M)")
    st.dataframe(_cf_stmt_df, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("### Balance Sheet (at Program Completion, $M)")
    st.dataframe(_bs_df, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("### Cumulative Cash Flow Over Time ($M)")
    st.altair_chart(_cf_chart.interactive(), use_container_width=True)
    st.caption(
        f"Investment phase: {program_start_quarter} {int(program_start_year)} → "
        f"{program_end_date.strftime('%b %Y')}. Revenue ramp: {_rev_ramp_years} years post-STC."
    )

with tab_sensitivity:
    st.subheader("Sensitivity Analysis — Impact on Net Revenue / Investment")
    st.caption("Each bar shows the ROI multiple when that input is varied ±20% from current value, holding all others constant.")

    def _calc_roi(
        mtow=None, eng_rate=None, mkt_pen=None, kit_price=None,
        cogs_pu=None, schedule=None, royalty=None,
    ):
        _mtow     = float(mtow      or mtow_lbs)
        _rate     = float(eng_rate  or eng_rate_usd_per_hr)
        _pen      = float(mkt_pen   or market_penetration_pct)
        _kp       = float(kit_price or kit_price_usd)
        _cogs     = float(cogs_pu   or cogs_per_unit_usd)
        _sched    = float(schedule  or schedule_months)
        _roy      = float(royalty   or royalty_pct)

        _rs       = _rate / 175.0
        _snre     = cal_df["total_nre_usd_m"].to_numpy() * _rs
        _a, _b    = fit_power_law(cal_df["mtow_lbs"].to_numpy(), _snre)
        _nre      = predict_power_law(_mtow, _a, _b) * 1e6
        _prod_r   = _nre * (float(prod_readiness_pct) / 100.0)

        if acquisition_mode == "Lease":
            _acq = lease_usd_per_month * _sched
        else:
            _acq = float(acquisition_cost_usd)

        _all_in   = _nre + _acq + float(tooling_usd) + _prod_r + float(inventory_usd) + float(flight_test_total_usd)
        _units    = float(fleet_size) * (_pen / 100.0)
        _gross    = _units * _kp
        _cogs_t   = _units * _cogs
        _roy_t    = _gross * (_roy / 100.0)
        _net      = _gross - _cogs_t - _roy_t - float(license_fee_usd)
        return _net / _all_in if _all_in > 0 else 0.0

    _base_roi = roi_multiple
    _delta = 0.20  # ±20%

    _sens_params = [
        ("Market penetration",   dict(mkt_pen=market_penetration_pct * (1 + _delta)),  dict(mkt_pen=market_penetration_pct * (1 - _delta))),
        ("Sales price/unit",     dict(kit_price=kit_price_usd * (1 + _delta)),          dict(kit_price=kit_price_usd * (1 - _delta))),
        ("COGS/unit",            dict(cogs_pu=cogs_per_unit_usd * (1 - _delta)),        dict(cogs_pu=cogs_per_unit_usd * (1 + _delta))),
        ("Engineering rate",     dict(eng_rate=eng_rate_usd_per_hr * (1 - _delta)),     dict(eng_rate=eng_rate_usd_per_hr * (1 + _delta))),
        ("Schedule months",      dict(schedule=schedule_months * (1 - _delta)),         dict(schedule=schedule_months * (1 + _delta))),
        ("Royalty rate",         dict(royalty=royalty_pct * (1 - _delta)),               dict(royalty=royalty_pct * (1 + _delta))),
        ("MTOW (NRE driver)",    dict(mtow=float(mtow_lbs) * (1 - _delta)),              dict(mtow=float(mtow_lbs) * (1 + _delta))),
    ]

    _tornado_rows = []
    for _label, _hi_kwargs, _lo_kwargs in _sens_params:
        _roi_hi = _calc_roi(**_hi_kwargs)
        _roi_lo = _calc_roi(**_lo_kwargs)
        _tornado_rows.append({
            "Input": _label,
            "ROI High": round(_roi_hi, 3),
            "ROI Low":  round(_roi_lo, 3),
            "Swing":    round(abs(_roi_hi - _roi_lo), 3),
        })

    _tornado_df = pd.DataFrame(_tornado_rows).sort_values("Swing", ascending=True)

    _t_long = pd.concat([
        _tornado_df[["Input", "ROI High"]].rename(columns={"ROI High": "ROI"}).assign(Scenario="+20%"),
        _tornado_df[["Input", "ROI Low"]].rename(columns={"ROI Low": "ROI"}).assign(Scenario="-20%"),
    ])

    _base_line = pd.DataFrame({"x": [_base_roi, _base_roi], "y_start": [0, len(_tornado_df)]})

    _tornado_chart = (
        alt.Chart(_t_long)
        .mark_bar(opacity=0.8)
        .encode(
            y=alt.Y("Input:N", sort=list(_tornado_df["Input"]), title="", axis=alt.Axis(labelLimit=200)),
            x=alt.X("ROI:Q", title="Net Revenue / Investment (x)", axis=alt.Axis(format=".1f")),
            color=alt.Color("Scenario:N", scale=alt.Scale(domain=["+20%", "-20%"], range=["#2ca02c", "#d62728"])),
            tooltip=[
                alt.Tooltip("Input:N"),
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("ROI:Q", title="ROI multiple", format=".2f"),
            ],
        )
    )

    _base_rule = (
        alt.Chart(pd.DataFrame({"x": [_base_roi]}))
        .mark_rule(color="black", strokeDash=[4, 4], strokeWidth=2)
        .encode(x=alt.X("x:Q"))
    )

    st.altair_chart(
        (_tornado_chart + _base_rule).properties(height=350).interactive(),
        use_container_width=True,
    )

    st.caption(f"Dashed line = base case ROI ({_base_roi:.2f}x). Bars show ROI at ±20% of each input.")

    st.divider()
    st.subheader("Break-Even Detail")
    _be_data = []
    for _pen_test in range(0, 101, 5):
        _u = float(fleet_size) * (_pen_test / 100.0)
        _g = _u * float(kit_price_usd)
        _n = _g - _u * float(cogs_per_unit_usd) - _g * (float(royalty_pct)/100.0) - float(license_fee_usd)
        _be_data.append({"Penetration (%)": _pen_test, "Cumulative Net Revenue ($M)": _n / 1e6})
    _be_df = pd.DataFrame(_be_data)
    _be_line = (
        alt.Chart(_be_df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("Penetration (%):Q", title="Market Penetration (%)"),
            y=alt.Y("Cumulative Net Revenue ($M):Q"),
            tooltip=["Penetration (%):Q", alt.Tooltip("Cumulative Net Revenue ($M):Q", format=".1f")],
        )
    )
    _cost_line = (
        alt.Chart(pd.DataFrame({"y": [all_in_cost_usd / 1e6]}))
        .mark_rule(color="#d62728", strokeDash=[4, 4])
        .encode(y=alt.Y("y:Q", title=""))
    )
    _zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="gray", strokeDash=[2, 2])
        .encode(y=alt.Y("y:Q"))
    )
    st.altair_chart(
        (_be_line + _cost_line + _zero_rule).properties(height=280).interactive(),
        use_container_width=True,
    )
    st.caption(
        f"Red dashed line = all-in program cost (\\${all_in_cost_usd/1e6:,.1f}M). "
        f"Break-even at ~{_be_pct:.1f}% penetration ({_be_units:,.0f} units)."
    )

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
            x=alt.X(
                "Start:T",
                title="Quarter",
                axis=alt.Axis(tickCount={"interval": "month", "step": 3}, format="%b %Y", labelAngle=-45),
            ),
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
