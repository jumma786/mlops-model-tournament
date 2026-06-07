# Chocolate Sales Analytics

> **Where should sales leadership focus reps, pricing, and SKU investment over the next two quarters to recover from the post-January revenue decline?**

A portfolio analytics project using Excel, SQL, and Power BI on 8 months of chocolate distribution data — 1,094 transactions, 6 countries, 25 sales reps, 22 products, ~$6.18M revenue.

---

## TL;DR — What the data says

| Finding | Evidence | Recommended action |
|---|---|---|
| **Canada has a pricing problem, not a volume problem** | Rev/box $30.84 vs. 6-country median ~$35.50. Volume is healthy at 31K boxes (2nd highest). | Audit Canada SKU mix and pricing. Closing half the gap = +$73K with no incremental volume. |
| **Revenue dropped 22% in February and never recovered** | Jan: $896K → Feb: $699K. June rebound to $865K was the only month within 4% of the January peak. | Run a Feb–Apr root-cause review before setting Q1-2023 targets off a depressed baseline. |
| **The bottom of the rep distribution is broken, not the top** | Top 5 reps cluster within 3% of each other ($311K–$321K). Bottom 5 span 45% ($138K–$202K). | Performance plan or territory reassignment for the bottom outlier (Wilone O'Kielt at $138K). |
| **USA is the highest-margin market and is under-served** | Rev/box $38.60 (11% above median), but 2nd-lowest box volume. | Reallocate 15% of shipping/rep capacity from Canada → USA. Estimated +$80K–$120K. |

---

## Project structure

```
├── Chocolate_Sales_Analytics.xlsx     ← Excel analysis with insights & charts
├── CHOCOLATES.sql                     ← 10 analytical queries (window functions, CTEs)
├── Sales_Performance_Dashboard.pbix   ← Interactive Power BI dashboard
└── README.md                          ← This file
```

---

## What's in the Excel workbook

| Sheet | What it contains |
|---|---|
| **README** | In-file project documentation |
| **Executive Summary** | KPI tiles, business question, headline findings |
| **Recommendations** | Prioritized action plan (P0/P1/P2) with evidence and impact estimates |
| **By Country** | Revenue, boxes, rev/box, share with bar chart |
| **By Salesperson** | All 25 reps ranked, with unit economics |
| **By Product** | All 22 SKUs ranked, with unit economics |
| **By Month** | Time series with MoM change and running total, line chart |
| **data** | 1,094 source transactions with calculated fields |

All aggregations use SUMIF formulas (not static pivots) so the workbook recalculates dynamically if data changes.

---

## What's in the SQL file

10 queries demonstrating MySQL 8.0+ window functions and CTEs. Each query opens with the **business question** it answers, not just the technique.

| # | Question | Technique |
|---|---|---|
| 1 | Who are the top performers? | `RANK()` over aggregated SUM |
| 2 | Top 3 products per country? | `PARTITION BY` + `DENSE_RANK` |
| 3 | Cumulative revenue tracking? | Running `SUM() OVER` |
| 4 | Where are the inflection points? | `LAG()` for prior-row access |
| 5 | How concentrated is rep contribution? | Nested aggregate `SUM(SUM()) OVER ()` |
| 6 | Which SKUs are most efficient? | `RANK` on `AVG(amount_per_box)` |
| 7 | How geographically diversified? | Country-share with nested aggregate |
| 8 | Best rep per country? | `PARTITION BY` + `DENSE_RANK = 1` |
| 9 | Workhorse vs. long-tail SKUs? | `NTILE(4)` quartile bucketing |
| 10 | Anomalous transactions? | 2-sigma statistical outlier detection |

---

## Methodology decisions worth flagging

**Why no valuation modeling here.** Sales transaction data describes operational performance — it does not give you the inputs needed for a defensible DCF (no balance sheet, no capital structure, no tax position, no FCF history). Mixing the two confuses the audience and weakens both stories. The valuation work for this project lives in a separate repository on a real public company.

**Why led with the business question.** A hiring manager reviewing a portfolio piece spends 60–90 seconds before forming a judgment. The first thing they should see is the question, the answer, and the evidence — not a list of tools. Tools are a means; a recommendation a sales director can act on Monday morning is the deliverable.

**Why three tools.** Each demonstrates a different competency:
- **Excel** — formula construction, layout, chart design, written communication
- **SQL** — window functions, CTEs, statistical methods
- **Power BI** — interactive exploration, drill-through, time intelligence

All three answer the same business question from different angles, which is realistic — most analytics roles use multiple tools on the same problem.

---

## How to read the analysis

1. Start with **Executive Summary** in the workbook for the headline findings.
2. Read **Recommendations** for the action plan and evidence.
3. The dimensional sheets (By Country / Salesperson / Product / Month) provide the supporting detail and charts.
4. Open **CHOCOLATES.sql** to see the same questions answered in SQL.
5. Open the **Power BI file** for interactive filtering and cross-highlighting.

---

## Tech stack

- **Excel** (Microsoft 365): SUMIF, RANK, native charts, conditional formatting
- **MySQL 8.0+**: Window functions (RANK, DENSE_RANK, NTILE, LAG, OVER), CTEs, statistical aggregates
- **Power BI Desktop**: DAX measures, interactive visuals

---

## About this dataset

Source: chocolate sales transaction file (Jan 3 – Aug 31, 2022). Data is anonymized but realistic — the rep names, countries, and product SKUs are the originals from the upstream dataset. Total revenue $6,183,625 across 177,007 boxes shipped.
