# Phase 2.4 — Backtest Results: Model vs. Actual Rating Actions

*Generated: 2026-02-19 08:16*

## 1. Executive Summary

- **Signal coverage:** 38/180 rating actions had at least one deterioration signal in the prior 180 days
- **Mean lead time:** 158 days
- **Median lead time:** 175 days
- **Lead time range:** 90 – 180 days

- **Best alert threshold** (by F1): N≥5 signals in 14-day window, 90-day lookahead
  - Precision: 79.0%, Recall: 73.2%, F1: 0.760

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
| DHFL | 2019-02-02 | watchlist_negative | CRISIL | 2018-11-04 | 90d | 118 / 134 |
| DHFL | 2019-02-03 | downgrade | CARE | 2018-11-04 | 91d | 122 / 138 |
| DHFL | 2019-02-03 | downgrade | ICRA | 2018-11-04 | 91d | 122 / 138 |
| DHFL | 2019-02-26 | downgrade | ICRA | 2018-11-04 | 114d | 178 / 197 |
| DHFL | 2019-02-27 | downgrade | CRISIL | 2018-11-04 | 115d | 181 / 200 |
| DHFL | 2019-04-17 | downgrade | CRISIL | 2018-11-04 | 164d | 238 / 270 |
| DHFL | 2019-05-01 | downgrade | CARE | 2018-11-04 | 178d | 246 / 278 |
| DHFL | 2019-05-13 | downgrade | CRISIL | 2018-11-14 | 180d | 246 / 278 |
| DHFL | 2019-05-14 | downgrade | CRISIL | 2018-11-19 | 176d | 251 / 283 |
| DHFL | 2019-05-14 | downgrade | CRISIL | 2018-11-19 | 176d | 251 / 283 |
| DHFL | 2019-05-14 | downgrade | ICRA | 2018-11-19 | 176d | 251 / 283 |
| DHFL | 2019-06-05 | default | CARE | 2018-12-11 | 176d | 312 / 344 |
| DHFL | 2019-06-05 | downgrade | CRISIL | 2018-12-11 | 176d | 312 / 344 |
| DHFL | 2019-06-05 | default | CRISIL | 2018-12-11 | 176d | 312 / 344 |
| DHFL | 2019-06-05 | default | ICRA | 2018-12-11 | 176d | 312 / 344 |
| DHFL | 2019-06-10 | downgrade | CRISIL | 2018-12-18 | 174d | 418 / 451 |
| DHFL | 2019-07-03 | default | ICRA | 2019-01-12 | 172d | 483 / 522 |
| DHFL | 2019-07-05 | default | CRISIL | 2019-01-12 | 174d | 489 / 528 |
| DHFL | 2019-07-10 | default | ICRA | 2019-01-12 | 179d | 493 / 532 |
| DHFL | 2019-10-18 | downgrade | CRISIL | 2019-04-22 | 179d | 511 / 543 |
| DHFL | 2019-12-20 | downgrade | CRISIL | 2019-06-24 | 179d | 512 / 542 |
| DHFL | 2019-12-20 | downgrade | CRISIL | 2019-06-24 | 179d | 512 / 542 |
| DHFL | 2020-01-28 | default | CRISIL | 2019-08-02 | 179d | 470 / 498 |
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
| Indiabulls Housing Finance | 2023-11-03 | downgrade | CRISIL | — | No signal | 0 / 0 |
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
| Reliance Capital | 2019-02-18 | outlook_negative | ICRA | 2018-11-20 | 90d | 103 / 146 |
| Reliance Capital | 2019-02-19 | outlook_negative | CRISIL | 2018-11-20 | 91d | 103 / 146 |
| Reliance Capital | 2019-03-05 | downgrade | ICRA | 2018-11-20 | 105d | 137 / 186 |
| Reliance Capital | 2019-04-19 | downgrade | CARE | 2018-11-20 | 150d | 174 / 226 |
| Reliance Capital | 2019-04-26 | downgrade | ICRA | 2018-11-20 | 157d | 182 / 234 |
| Reliance Capital | 2019-05-19 | downgrade | CARE | 2018-11-20 | 180d | 237 / 292 |
| Reliance Capital | 2019-06-21 | downgrade | CARE | 2018-12-28 | 175d | 307 / 379 |
| Reliance Capital | 2019-06-21 | downgrade | CRISIL | 2018-12-28 | 175d | 307 / 379 |
| Reliance Capital | 2019-06-21 | downgrade | ICRA | 2018-12-28 | 175d | 307 / 379 |
| Reliance Capital | 2019-09-20 | default | CARE | 2019-03-30 | 174d | 287 / 356 |
| Reliance Capital | 2019-10-01 | downgrade | CRISIL | 2019-04-18 | 166d | 344 / 425 |
| Reliance Capital | 2019-10-01 | downgrade | ICRA | 2019-04-18 | 166d | 344 / 425 |
| Reliance Capital | 2019-11-15 | default | CARE | 2019-05-19 | 180d | 317 / 400 |
| Reliance Capital | 2019-11-15 | default | CRISIL | 2019-05-19 | 180d | 317 / 400 |
| Reliance Capital | 2019-11-15 | default | ICRA | 2019-05-19 | 180d | 317 / 400 |
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
| Spandana Sphoorty | 2024-07-15 | outlook_negative | ICRA | — | No signal | 0 / 0 |
| Spandana Sphoorty | 2025-02-03 | downgrade | ICRA | — | No signal | 0 / 0 |
| Spandana Sphoorty | 2025-02-03 | downgrade | ICRA | — | No signal | 0 / 0 |
| Spandana Sphoorty | 2025-02-10 | downgrade | CRISIL | — | No signal | 0 / 0 |
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

- **Articles:** 1243 (2018-11-04 to 2022-04-13)
- **Deterioration signals:** 1152 (92.7%)
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

**Signal-before-action coverage:** 23/23
**Mean lead time:** 160 days
**Earliest signal:** 2018-11-04

## 3. Reliance Capital Deep Dive

- **Articles:** 688 (2018-11-20 to 2019-11-15)
- **Deterioration signals:** 550 (79.9%)
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

**Signal-before-action coverage:** 15/15
**Mean lead time:** 156 days
**Earliest signal:** 2018-11-20

## 3. Cholamandalam Deep Dive

- **Articles:** 1372 (2017-06-09 to 2024-12-24)
- **Deterioration signals:** 179 (13.0%)
- **Rating actions:** 0 (clean record)

**False positive rate:** 179/1372 articles (13.0%) predicted as deterioration
Cholamandalam has zero downgrades — every deterioration signal is a false positive.

## 4. Alert Threshold Grid

Grid search over N (min signals), M (window days), K (lookahead days).

| N | Window | Lookahead | Alerts | TP | FP | FN | Precision | Recall | F1 |
|---|--------|-----------|--------|----|----|-----|-----------|--------|----|
| 5 | 14d | 180d | 490 | 387 | 103 | 142 | 79.0% | 73.2% | 0.760 |
| 5 | 14d | 90d | 490 | 387 | 103 | 142 | 79.0% | 73.2% | 0.760 |
| 5 | 30d | 180d | 602 | 400 | 202 | 142 | 66.4% | 73.8% | 0.699 |
| 5 | 30d | 90d | 602 | 400 | 202 | 142 | 66.4% | 73.8% | 0.699 |
| 3 | 14d | 90d | 602 | 394 | 208 | 142 | 65.4% | 73.5% | 0.692 |
| 3 | 14d | 180d | 602 | 394 | 208 | 142 | 65.4% | 73.5% | 0.692 |
| 2 | 14d | 90d | 707 | 400 | 307 | 142 | 56.6% | 73.8% | 0.641 |
| 2 | 14d | 180d | 707 | 400 | 307 | 142 | 56.6% | 73.8% | 0.641 |
| 5 | 60d | 180d | 741 | 406 | 335 | 142 | 54.8% | 74.1% | 0.630 |
| 5 | 60d | 90d | 741 | 406 | 335 | 142 | 54.8% | 74.1% | 0.630 |
| 3 | 30d | 180d | 762 | 406 | 356 | 142 | 53.3% | 74.1% | 0.620 |
| 3 | 30d | 90d | 762 | 406 | 356 | 142 | 53.3% | 74.1% | 0.620 |
| 2 | 30d | 90d | 874 | 407 | 467 | 142 | 46.6% | 74.1% | 0.572 |
| 2 | 30d | 180d | 874 | 407 | 467 | 142 | 46.6% | 74.1% | 0.572 |
| 5 | 90d | 90d | 873 | 406 | 467 | 142 | 46.5% | 74.1% | 0.571 |
| 5 | 90d | 180d | 873 | 406 | 467 | 142 | 46.5% | 74.1% | 0.571 |
| 1 | 14d | 180d | 894 | 410 | 484 | 142 | 45.9% | 74.3% | 0.567 |
| 1 | 14d | 90d | 894 | 410 | 484 | 142 | 45.9% | 74.3% | 0.567 |
| 3 | 60d | 90d | 928 | 407 | 521 | 142 | 43.9% | 74.1% | 0.551 |
| 3 | 60d | 180d | 928 | 407 | 521 | 142 | 43.9% | 74.1% | 0.551 |
| 2 | 60d | 180d | 999 | 407 | 592 | 142 | 40.7% | 74.1% | 0.526 |
| 2 | 60d | 90d | 999 | 407 | 592 | 142 | 40.7% | 74.1% | 0.526 |
| 1 | 30d | 180d | 1022 | 411 | 611 | 142 | 40.2% | 74.3% | 0.522 |
| 1 | 30d | 90d | 1022 | 411 | 611 | 142 | 40.2% | 74.3% | 0.522 |
| 3 | 90d | 90d | 1039 | 407 | 632 | 142 | 39.2% | 74.1% | 0.513 |
| 3 | 90d | 180d | 1039 | 407 | 632 | 142 | 39.2% | 74.1% | 0.513 |
| 2 | 90d | 180d | 1081 | 407 | 674 | 142 | 37.7% | 74.1% | 0.499 |
| 2 | 90d | 90d | 1081 | 407 | 674 | 142 | 37.7% | 74.1% | 0.499 |
| 1 | 60d | 180d | 1139 | 411 | 728 | 142 | 36.1% | 74.3% | 0.486 |
| 1 | 60d | 90d | 1139 | 411 | 728 | 142 | 36.1% | 74.3% | 0.486 |
| 1 | 90d | 180d | 1187 | 411 | 776 | 142 | 34.6% | 74.3% | 0.472 |
| 1 | 90d | 90d | 1187 | 411 | 776 | 142 | 34.6% | 74.3% | 0.472 |

## 5. Naive Baselines

| Baseline | Description | Signal Coverage | Mean Lead Time | Median Lead Time |
|----------|-------------|----------------|----------------|-----------------|
| **Our Model** | LoRA fine-tuned Qwen 2.5-7B | 38/180 | 158d | 175d |
| always_deterioration | Predict deterioration for every article | 38/180 | 159d | 176d |
| ground_truth_labels | Use Haiku/Sonnet labels as predictions | 38/180 | 157d | 175d |

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
