# Data directory

This directory includes the anonymized, numerically coded analysis dataset:

```text
survey_data.xlsx
```

The file contains 361 rows, one per valid response, and 50 coded variables. The modeling workflow uses the following columns:

| Column | Description | Expected values |
|---|---|---|
| `Satisfaction_Overall` | Overall satisfaction | Integer, 1–5 |
| `Satisfaction_Access` | Accessibility satisfaction | Integer, 1–5 |
| `Satisfaction_Comfort` | Comfort satisfaction | Integer, 1–5 |
| `Satisfaction_Cleanliness` | Cleanliness satisfaction | Integer, 1–5 |
| `Satisfaction_Safety` | Safety satisfaction | Integer, 1–5 |
| `Satisfaction_Greenery` | Greenery satisfaction | Integer, 1–5 |
| `Satisfaction_Maintenance` | Maintenance satisfaction | Integer, 1–5 |
| `Facility_Seating` | Seating rating | Integer, 1–5 |
| `Facility_Fitness` | Fitness-facility rating | Integer, 1–5 |
| `Facility_Children` | Children's-facility rating | Integer, 1–5 |
| `Facility_Shade` | Shade-facility rating | Integer, 1–5 |
| `Facility_WiFi` | Wi-Fi rating | Integer, 1–5 |

The released workbook contains no names, contact details, account identifiers, IP addresses, precise timestamps, precise addresses, free-text responses, formulas, comments, hyperlinks, or hidden worksheets. Workbook creator metadata was removed during preparation. The questionnaire form is not included.

If a recommendation-score variable is added, document its stored and analytical scales explicitly. A response stored as 1–11 for an original 0–10 scale should be recoded as:

```text
NPS_Score_0_10 = NPS_Score_raw - 1
```
