# Analysis data

`survey_data.xlsx` contains 361 valid responses and 50 numerically coded variables.
Cell values are unchanged from the previously released workbook.

The satisfaction models use `Satisfaction_Overall` as the outcome and these eleven
predictors (all rating items use integers 1 through 5):

- `Satisfaction_Access`, `Satisfaction_Comfort`, `Satisfaction_Cleanliness`,
  `Satisfaction_Safety`, `Satisfaction_Greenery`, `Satisfaction_Maintenance`
- `Facility_Seating`, `Facility_Fitness`, `Facility_Children`, `Facility_Shade`,
  `Facility_WiFi`

EFA and clustering use all six `Satisfaction_` attributes (excluding the overall
item), eight `Facility_` items, and three `Impact_` items, for 17 items in total.

Age uses six categories: 1=18-25, 2=26-35, 3=36-45, 4=46-55, 5=56-65, 6=66+.
The stored recommendation score ranges from 1 to 11. The audit subtracts one to
represent 0 through 10. Its near-zero association with satisfaction is unresolved;
it is not used in the revised satisfaction models or substantive conclusions.

The workbook excludes names, contact details, account identifiers, IP addresses,
precise addresses, timestamps, and free-text responses. It contains no formulas,
comments, hyperlinks, or hidden worksheets. Creator metadata was removed during
preparation. The questionnaire and ethics documents are not included.

The data do not contain park identifiers or raw recruitment/exclusion records.
The recruitment denominator, exclusion reasons, geographic screening, and consent
interface cannot be inferred from this workbook.
