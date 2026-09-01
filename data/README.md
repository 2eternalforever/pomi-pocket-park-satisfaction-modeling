# Data directory

The participant-level dataset is not included. Place an anonymized and sharing-approved Excel file here as:

```text
survey_data.xlsx
```

The file must contain one row per response and these columns:

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

Before sharing participant-level data, remove direct identifiers, account identifiers, IP addresses, precise timestamps, precise addresses, free-text identifiers, and any other fields that could reasonably enable re-identification. Confirm that the consent and ethics terms permit the intended form of data sharing.

If a recommendation-score variable is added, document its stored and analytical scales explicitly. A response stored as 1–11 for an original 0–10 scale should be recoded as:

```text
NPS_Score_0_10 = NPS_Score_raw - 1
```
