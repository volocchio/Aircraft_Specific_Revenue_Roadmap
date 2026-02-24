# Tamarack STC Cost & Schedule Estimator

Streamlit app for estimating certification cost and schedule for Tamarack winglet STCs across aircraft types.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key inputs

| Input | Description |
|---|---|
| Aircraft / MTOW | Pick from list or enter custom MTOW (lbs) |
| Acquisition mode | Lease (cost/month × months held) or Purchase (price minus resale) |
| Tooling | Non-recurring tooling cost |
| Production readiness | Pre-production / process engineering cost |
| Inventory | Parts inventory needed to begin installations at STC/PC/145 award |
| Schedule | Estimated months to STC |

## How costs are estimated

`Base certification NRE` is driven by a **power-law fit** through historical MTOW → Total NRE data points stored in `data/nre_calibration.csv`.  
All other line items (acquisition, tooling, production readiness, inventory) are entered directly.

## Adding a new aircraft

Edit `data/aircraft.csv` — add a row with `aircraft`, `mtow_lbs`, and optional `notes`.

## Adding a new calibration point

Edit `data/nre_calibration.csv` — add a row with `mtow_lbs` and `total_nre_usd_m`.  
The power-law fit is recomputed automatically on every app load.

## Reference programs

Drop completed program actuals, proposals, or data files into `reference/` — the app will pick up new calibration points when you update `data/nre_calibration.csv`.
