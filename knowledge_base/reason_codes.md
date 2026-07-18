# Adverse action reason codes

Approved language for communicating declines to applicants.

| Code | Applicant-facing reason | Triggering features |
|------|------------------------|--------------------|
| AA01 | Insufficient credit history with external bureaus | EXT_SOURCE_1/2/3 missing or low |
| AA02 | Record of late payments on existing or previous credit | INST_IS_LATE_MEAN, INST_DAYS_LATE_MAX |
| AA03 | Amounts paid were less than amounts due on prior credit | INST_PAYMENT_DIFF_MEAN |
| AA04 | Level of existing debt relative to income | ANNUITY_INCOME_RATIO |
| AA05 | Amount of credit requested relative to income | CREDIT_INCOME_RATIO |
| AA06 | Overdue balances reported at other institutions | BUREAU_OVERDUE_COUNT |
| AA07 | Number of credit obligations currently outstanding | BUREAU_ACTIVE_COUNT |
| AA08 | Length of employment history | YEARS_EMPLOYED, DAYS_EMPLOYED_ANOM |
| AA09 | Previous applications for credit were not approved | PREV_REFUSAL_RATE |
| AA10 | Requested repayment structure relative to loan size | CREDIT_TERM |

## Guidance on use - for Customer Care Agents

State the two to four factors that most influenced the decision, ordered by contribution.
Do not state that the decision was made by a computer without human involvement. Do not
disclose model coefficients, feature weights, or internal thresholds. Use plain language;
avoid internal feature names in applicant-facing text.