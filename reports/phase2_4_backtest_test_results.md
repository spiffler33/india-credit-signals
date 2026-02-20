# Phase 2.4 — Backtest Results: Model vs. Actual Rating Actions

*Generated: 2026-02-19 08:17*

## 1. Executive Summary

- **Signal coverage:** 5/180 rating actions had at least one deterioration signal in the prior 180 days
- **Mean lead time:** 119 days
- **Median lead time:** 166 days
- **Lead time range:** 21 – 173 days

- **Best alert threshold** (by F1): N≥1 signals in 60-day window, 180-day lookahead
  - Precision: 4.6%, Recall: 6.9%, F1: 0.056

## 2. Lead Time Analysis

For each rating action (downgrade/default), we looked back 180 days for deterioration signals from the model.

| Entity | Action Date | Type | Agency | First Signal | Lead Time | Signals |
|--------|------------|------|--------|-------------|-----------|---------|
| Altico Capital | 2019-09-03 | downgrade | India Ratings | — | No signal | 0 / 0 |
| Altico Capital | 2019-09-03 | downgrade | India Ratings | — | No signal | 0 / 0 |
| Altico Capital | 2019-09-13 | downgrade | CARE | — | No signal | 0 / 0 |
| Altico Capital | 2019-09-13 | default | India Ratings | — | No signal | 0 / 0 |
| Altico Capital | 2019-09-13 | default | India Ratings | — | No signal | 0 / 0 |
| Can Fin Homes | 2019-05-06 | downgrade | ICRA | — | No signal | 0 / 0 |
| CreditAccess Grameen | 2021-08-10 | downgrade | ICRA | — | No signal | 0 / 0 |
| CreditAccess Grameen | 2021-12-24 | downgrade | ICRA | — | No signal | 0 / 0 |
| DHFL | 2019-02-02 | watchlist_negative | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-02-03 | downgrade | CARE | — | No signal | 0 / 0 |
| DHFL | 2019-02-03 | downgrade | ICRA | — | No signal | 0 / 0 |
| DHFL | 2019-02-26 | downgrade | ICRA | — | No signal | 0 / 0 |
| DHFL | 2019-02-27 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-04-17 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-05-01 | downgrade | CARE | — | No signal | 0 / 0 |
| DHFL | 2019-05-13 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-05-14 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-05-14 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-05-14 | downgrade | ICRA | — | No signal | 0 / 0 |
| DHFL | 2019-06-05 | default | CARE | — | No signal | 0 / 0 |
| DHFL | 2019-06-05 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-06-05 | default | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-06-05 | default | ICRA | — | No signal | 0 / 0 |
| DHFL | 2019-06-10 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-07-03 | default | ICRA | — | No signal | 0 / 0 |
| DHFL | 2019-07-05 | default | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-07-10 | default | ICRA | — | No signal | 0 / 0 |
| DHFL | 2019-10-18 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-12-20 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2019-12-20 | downgrade | CRISIL | — | No signal | 0 / 0 |
| DHFL | 2020-01-28 | default | CRISIL | — | No signal | 0 / 0 |
| Fusion Micro Finance | 2014-12-09 | downgrade | ICRA | — | No signal | 0 / 0 |
| Fusion Micro Finance | 2014-12-31 | downgrade | ICRA | — | No signal | 0 / 0 |
| Fusion Micro Finance | 2015-04-27 | downgrade | ICRA | — | No signal | 0 / 0 |
| Fusion Micro Finance | 2015-07-03 | downgrade | ICRA | — | No signal | 0 / 0 |
| Fusion Micro Finance | 2015-07-07 | downgrade | ICRA | — | No signal | 0 / 0 |
| Fusion Micro Finance | 2016-06-02 | downgrade | ICRA | — | No signal | 0 / 0 |
| Fusion Micro Finance | 2016-06-06 | downgrade | ICRA | — | No signal | 0 / 0 |
| Fusion Micro Finance | 2016-10-20 | downgrade | ICRA | — | No signal | 0 / 0 |
| Fusion Micro Finance | 2017-03-21 | downgrade | ICRA | — | No signal | 0 / 0 |
| HDFC Ltd | 2018-01-18 | downgrade | ICRA | — | No signal | 0 / 0 |
| HDFC Ltd | 2019-05-17 | downgrade | ICRA | — | No signal | 0 / 0 |
| IL&FS | 2018-08-06 | downgrade | ICRA | — | No signal | 0 / 0 |
| IL&FS | 2018-08-06 | downgrade | ICRA | — | No signal | 0 / 0 |
| IL&FS | 2018-09-08 | downgrade | CARE | — | No signal | 0 / 0 |
| IL&FS | 2018-09-08 | downgrade | CARE | — | No signal | 0 / 0 |
| IL&FS | 2018-09-08 | downgrade | ICRA | — | No signal | 0 / 0 |
| IL&FS | 2018-09-08 | downgrade | ICRA | — | No signal | 0 / 0 |
| IL&FS | 2018-09-08 | downgrade | ICRA | — | No signal | 0 / 0 |
| IL&FS | 2018-09-08 | downgrade | India Ratings | — | No signal | 0 / 0 |
| IL&FS | 2018-09-08 | downgrade | India Ratings | — | No signal | 0 / 0 |
| IL&FS | 2018-09-17 | default | CARE | — | No signal | 0 / 0 |
| IL&FS | 2018-09-17 | default | CARE | — | No signal | 0 / 0 |
| IL&FS | 2018-09-17 | default | CRISIL | — | No signal | 0 / 0 |
| IL&FS | 2018-09-17 | default | ICRA | — | No signal | 0 / 0 |
| IL&FS | 2018-09-17 | default | ICRA | — | No signal | 0 / 0 |
| IL&FS | 2018-09-17 | default | ICRA | — | No signal | 0 / 0 |
| IL&FS | 2018-09-17 | default | India Ratings | — | No signal | 0 / 0 |
| IL&FS Financial Services | 2018-08-16 | downgrade | CARE | — | No signal | 0 / 0 |
| IL&FS Financial Services | 2018-09-08 | downgrade | CARE | — | No signal | 0 / 0 |
| IL&FS Financial Services | 2018-09-08 | downgrade | ICRA | — | No signal | 0 / 0 |
| IL&FS Financial Services | 2018-09-08 | downgrade | ICRA | — | No signal | 0 / 0 |
| IL&FS Financial Services | 2018-09-08 | downgrade | India Ratings | — | No signal | 0 / 0 |
| IL&FS Financial Services | 2018-09-17 | default | CARE | — | No signal | 0 / 0 |
| IL&FS Financial Services | 2018-09-17 | default | ICRA | — | No signal | 0 / 0 |
| IL&FS Financial Services | 2018-09-17 | default | India Ratings | — | No signal | 0 / 0 |
| Indiabulls Housing Finance | 2019-08-30 | downgrade | ICRA | — | No signal | 0 / 0 |
| Indiabulls Housing Finance | 2019-09-10 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Indiabulls Housing Finance | 2020-02-07 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Indiabulls Housing Finance | 2020-02-20 | downgrade | ICRA | — | No signal | 0 / 0 |
| Indiabulls Housing Finance | 2023-11-03 | downgrade | CRISIL | 2023-08-28 | 67d | 1 / 8 |
| Lakshmi Vilas Bank | 2019-06-25 | outlook_negative | ICRA | — | No signal | 0 / 0 |
| Lakshmi Vilas Bank | 2019-09-27 | downgrade | CARE | — | No signal | 0 / 0 |
| Lakshmi Vilas Bank | 2019-09-27 | downgrade | ICRA | — | No signal | 0 / 0 |
| Lakshmi Vilas Bank | 2019-10-09 | downgrade | Brickwork | — | No signal | 0 / 0 |
| Lakshmi Vilas Bank | 2020-09-29 | downgrade | ICRA | — | No signal | 0 / 0 |
| Lakshmi Vilas Bank | 2020-10-07 | downgrade | Brickwork | — | No signal | 0 / 0 |
| Lakshmi Vilas Bank | 2020-10-09 | downgrade | CARE | — | No signal | 0 / 0 |
| Mahindra Finance | 2009-03-19 | downgrade | CRISIL | — | No signal | 0 / 0 |
| PFC | 2015-09-24 | default | ICRA | — | No signal | 0 / 0 |
| PFC | 2015-09-25 | default | CRISIL | — | No signal | 0 / 0 |
| PMC Bank | 2019-09-24 | default | CARE | — | No signal | 0 / 0 |
| PNB Housing Finance | 2010-02-10 | downgrade | CRISIL | — | No signal | 0 / 0 |
| PNB Housing Finance | 2016-03-01 | downgrade | ICRA | — | No signal | 0 / 0 |
| PNB Housing Finance | 2020-02-21 | downgrade | CRISIL | — | No signal | 0 / 0 |
| PNB Housing Finance | 2020-04-03 | downgrade | ICRA | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-05-07 | downgrade | ICRA | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-05-14 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-05-14 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-05-14 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-05-14 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-05-14 | downgrade | ICRA | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-06-10 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-06-10 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-10-14 | downgrade | ICRA | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-10-14 | default | ICRA | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-10-18 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-12-13 | default | ICRA | — | No signal | 0 / 0 |
| Piramal Enterprises | 2019-12-20 | downgrade | CRISIL | — | No signal | 0 / 0 |
| REC | 2019-02-25 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Capital | 2019-02-18 | outlook_negative | ICRA | — | No signal | 0 / 0 |
| Reliance Capital | 2019-02-19 | outlook_negative | CRISIL | — | No signal | 0 / 0 |
| Reliance Capital | 2019-03-05 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Capital | 2019-04-19 | downgrade | CARE | — | No signal | 0 / 0 |
| Reliance Capital | 2019-04-26 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Capital | 2019-05-19 | downgrade | CARE | — | No signal | 0 / 0 |
| Reliance Capital | 2019-06-21 | downgrade | CARE | — | No signal | 0 / 0 |
| Reliance Capital | 2019-06-21 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Reliance Capital | 2019-06-21 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Capital | 2019-09-20 | default | CARE | — | No signal | 0 / 0 |
| Reliance Capital | 2019-10-01 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Reliance Capital | 2019-10-01 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Capital | 2019-11-15 | default | CARE | — | No signal | 0 / 0 |
| Reliance Capital | 2019-11-15 | default | CRISIL | — | No signal | 0 / 0 |
| Reliance Capital | 2019-11-15 | default | ICRA | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-04-03 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-04-20 | downgrade | CARE | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-04-26 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-04-30 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-04-30 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-06-21 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-06-21 | downgrade | ICRA | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-07-01 | default | CARE | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-10-01 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-11-15 | default | CRISIL | — | No signal | 0 / 0 |
| Reliance Home Finance | 2019-11-15 | default | ICRA | — | No signal | 0 / 0 |
| SREI Equipment Finance | 2019-10-10 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Equipment Finance | 2020-06-03 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Equipment Finance | 2020-11-19 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Equipment Finance | 2021-01-22 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Equipment Finance | 2021-03-16 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Equipment Finance | 2021-03-16 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Equipment Finance | 2021-07-22 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Equipment Finance | 2021-10-13 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Equipment Finance | 2021-10-13 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2019-07-01 | outlook_negative | Brickwork | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2019-09-10 | downgrade | Brickwork | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2020-06-30 | outlook_negative | ICRA | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2020-10-28 | downgrade | ICRA | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2020-11-19 | downgrade | CRISIL | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2020-11-24 | downgrade | Brickwork | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2021-01-15 | downgrade | Acuite | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2021-03-05 | default | Acuite | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2021-03-06 | default | CARE | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2021-03-08 | downgrade | ICRA | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2021-10-04 | default | CARE | — | No signal | 0 / 0 |
| SREI Infrastructure Finance | 2021-10-04 | default | ICRA | — | No signal | 0 / 0 |
| Shriram Finance | 2020-06-23 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Spandana Sphoorty | 2023-08-18 | outlook_negative | ICRA | — | No signal | 0 / 0 |
| Spandana Sphoorty | 2024-07-15 | outlook_negative | ICRA | 2024-06-24 | 21d | 1 / 4 |
| Spandana Sphoorty | 2025-02-03 | downgrade | ICRA | 2024-08-21 | 166d | 4 / 9 |
| Spandana Sphoorty | 2025-02-03 | downgrade | ICRA | 2024-08-21 | 166d | 4 / 9 |
| Spandana Sphoorty | 2025-02-10 | downgrade | CRISIL | 2024-08-21 | 173d | 4 / 9 |
| Spandana Sphoorty | 2025-06-11 | downgrade | ICRA | — | No signal | 0 / 0 |
| Spandana Sphoorty | 2025-06-11 | downgrade | ICRA | — | No signal | 0 / 0 |
| Spandana Sphoorty | 2025-07-01 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Spandana Sphoorty | 2025-08-26 | downgrade | ICRA | — | No signal | 0 / 0 |
| Spandana Sphoorty | 2025-08-26 | downgrade | ICRA | — | No signal | 0 / 0 |
| Tata Capital | 2019-08-14 | downgrade | CRISIL | — | No signal | 0 / 0 |
| Tata Capital | 2019-08-14 | downgrade | CRISIL | — | No signal | 0 / 0 |
| YES Bank | 2015-12-31 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2018-11-28 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2018-11-28 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2019-05-03 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2019-05-03 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2019-05-03 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2019-07-24 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2019-07-24 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2019-07-24 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2019-12-19 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2019-12-19 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2019-12-19 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2020-02-20 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2020-02-20 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2020-02-20 | downgrade | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2020-03-06 | default | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2020-03-06 | default | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2020-03-06 | default | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2020-06-23 | default | ICRA | — | No signal | 0 / 0 |
| YES Bank | 2021-09-09 | default | ICRA | — | No signal | 0 / 0 |

## 3. DHFL Deep Dive

- **Rating actions:** 23
  - 2019-02-02: watchlist_negative (CRISIL)
  - 2019-02-03: downgrade (CARE)
  - 2019-02-03: downgrade (ICRA)
  - 2019-02-26: downgrade (ICRA)
  - 2019-02-27: downgrade (CRISIL)
  - 2019-04-17: downgrade (CRISIL)
  - 2019-05-01: downgrade (CARE)
  - 2019-05-13: downgrade (CRISIL)
  - 2019-05-14: downgrade (CRISIL)
  - 2019-05-14: downgrade (CRISIL)
  - 2019-05-14: downgrade (ICRA)
  - 2019-06-05: default (ICRA)
  - 2019-06-05: default (CRISIL)
  - 2019-06-05: default (CARE)
  - 2019-06-05: downgrade (CRISIL)
  - 2019-06-10: downgrade (CRISIL)
  - 2019-07-03: default (ICRA)
  - 2019-07-05: default (CRISIL)
  - 2019-07-10: default (ICRA)
  - 2019-10-18: downgrade (CRISIL)
  - 2019-12-20: downgrade (CRISIL)
  - 2019-12-20: downgrade (CRISIL)
  - 2020-01-28: default (CRISIL)

**No deterioration signals detected before rating actions.**

## 3. Reliance Capital Deep Dive

- **Rating actions:** 15
  - 2019-02-18: outlook_negative (ICRA)
  - 2019-02-19: outlook_negative (CRISIL)
  - 2019-03-05: downgrade (ICRA)
  - 2019-04-19: downgrade (CARE)
  - 2019-04-26: downgrade (ICRA)
  - 2019-05-19: downgrade (CARE)
  - 2019-06-21: downgrade (CARE)
  - 2019-06-21: downgrade (CRISIL)
  - 2019-06-21: downgrade (ICRA)
  - 2019-09-20: default (CARE)
  - 2019-10-01: downgrade (CRISIL)
  - 2019-10-01: downgrade (ICRA)
  - 2019-11-15: default (CARE)
  - 2019-11-15: default (CRISIL)
  - 2019-11-15: default (ICRA)

**No deterioration signals detected before rating actions.**

## 3. Cholamandalam Deep Dive

- **Rating actions:** 0 (clean record)

No predictions for Cholamandalam in this dataset.

## 4. Alert Threshold Grid

Grid search over N (min signals), M (window days), K (lookahead days).

| N | Window | Lookahead | Alerts | TP | FP | FN | Precision | Recall | F1 |
|---|--------|-----------|--------|----|----|-----|-----------|--------|----|
| 1 | 60d | 180d | 280 | 13 | 267 | 175 | 4.6% | 6.9% | 0.056 |
| 1 | 90d | 180d | 291 | 13 | 278 | 175 | 4.5% | 6.9% | 0.054 |
| 1 | 14d | 180d | 194 | 9 | 185 | 175 | 4.6% | 4.9% | 0.048 |
| 1 | 30d | 180d | 229 | 9 | 220 | 175 | 3.9% | 4.9% | 0.044 |
| 3 | 90d | 180d | 176 | 6 | 170 | 177 | 3.4% | 3.3% | 0.033 |
| 1 | 60d | 90d | 280 | 7 | 273 | 178 | 2.5% | 3.8% | 0.030 |
| 1 | 90d | 90d | 291 | 7 | 284 | 178 | 2.4% | 3.8% | 0.029 |
| 2 | 90d | 180d | 228 | 6 | 222 | 177 | 2.6% | 3.3% | 0.029 |
| 1 | 14d | 90d | 194 | 5 | 189 | 178 | 2.6% | 2.7% | 0.027 |
| 2 | 60d | 180d | 205 | 5 | 200 | 177 | 2.4% | 2.7% | 0.026 |
| 3 | 60d | 180d | 142 | 4 | 138 | 177 | 2.8% | 2.2% | 0.025 |
| 1 | 30d | 90d | 229 | 5 | 224 | 178 | 2.2% | 2.7% | 0.024 |
| 2 | 30d | 180d | 151 | 4 | 147 | 177 | 2.6% | 2.2% | 0.024 |
| 3 | 30d | 180d | 83 | 3 | 80 | 177 | 3.6% | 1.7% | 0.023 |
| 2 | 14d | 180d | 101 | 2 | 99 | 177 | 2.0% | 1.1% | 0.014 |
| 3 | 14d | 180d | 44 | 1 | 43 | 177 | 2.3% | 0.6% | 0.009 |
| 5 | 60d | 90d | 64 | 0 | 64 | 180 | 0.0% | 0.0% | 0.000 |
| 5 | 30d | 180d | 21 | 0 | 21 | 180 | 0.0% | 0.0% | 0.000 |
| 5 | 60d | 180d | 64 | 0 | 64 | 180 | 0.0% | 0.0% | 0.000 |
| 5 | 30d | 90d | 21 | 0 | 21 | 180 | 0.0% | 0.0% | 0.000 |
| 5 | 14d | 180d | 10 | 0 | 10 | 180 | 0.0% | 0.0% | 0.000 |
| 5 | 90d | 90d | 95 | 0 | 95 | 180 | 0.0% | 0.0% | 0.000 |
| 5 | 14d | 90d | 10 | 0 | 10 | 180 | 0.0% | 0.0% | 0.000 |
| 3 | 14d | 90d | 44 | 0 | 44 | 180 | 0.0% | 0.0% | 0.000 |
| 3 | 90d | 90d | 176 | 0 | 176 | 180 | 0.0% | 0.0% | 0.000 |
| 3 | 60d | 90d | 142 | 0 | 142 | 180 | 0.0% | 0.0% | 0.000 |
| 3 | 30d | 90d | 83 | 0 | 83 | 180 | 0.0% | 0.0% | 0.000 |
| 2 | 90d | 90d | 228 | 0 | 228 | 180 | 0.0% | 0.0% | 0.000 |
| 2 | 60d | 90d | 205 | 0 | 205 | 180 | 0.0% | 0.0% | 0.000 |
| 2 | 30d | 90d | 151 | 0 | 151 | 180 | 0.0% | 0.0% | 0.000 |
| 2 | 14d | 90d | 101 | 0 | 101 | 180 | 0.0% | 0.0% | 0.000 |
| 5 | 90d | 180d | 95 | 0 | 95 | 180 | 0.0% | 0.0% | 0.000 |

## 5. Naive Baselines

| Baseline | Description | Signal Coverage | Mean Lead Time | Median Lead Time |
|----------|-------------|----------------|----------------|-----------------|
| **Our Model** | LoRA fine-tuned Qwen 2.5-7B | 5/180 | 119d | 166d |
| always_deterioration | Predict deterioration for every article | 5/180 | 134d | 166d |
| ground_truth_labels | Use Haiku/Sonnet labels as predictions | 5/180 | 120d | 166d |

## 6. Limitations & Next Steps

### Limitations
- **Small event set:** Only 2-3 entities with downgrades in holdout, limiting statistical power
- **Label quality:** Training data labels from Haiku (82.3% agreement with Sonnet) — model inherits labeling biases
- **Survivorship bias:** Articles sourced from GDELT may not capture all relevant news
- **Entity overlap:** Multi-entity articles mentioning holdout entities stayed in training

### Next Steps
- **Prompted Opus baseline:** Run Claude Opus on holdout articles for apples-to-apples comparison
- **FinRLlama baseline:** Run FinRLlama 3.2-3B on holdout articles (requires Colab)
- **Contagion analysis (Phase 3):** Does DHFL crisis signal propagate to other NBFCs?
- **Dashboard (Phase 4):** Visualize timelines with interactive plots
