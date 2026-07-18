# Feature data dictionary

## External bureau scores
**EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3** — Normalised credit scores supplied by
external credit bureaus, scaled 0 to 1. Higher values indicate lower assessed risk.
These are the single strongest predictors in the model. A missing value is itself
informative and may indicate a thin credit file.

## Payment behaviour features
**INST_IS_LATE_MEAN** — Proportion of the applicant's historical instalments that were
paid after the due date. Range 0 to 1. Values above 0.30 indicate a persistent pattern
of late payment and are a strong risk signal.

**INST_PAYMENT_DIFF_MEAN** — Average shortfall between the amount due and the amount
actually paid. Positive values mean the applicant systematically underpays.

**INST_DAYS_LATE_MAX** — The single worst payment delay on record, in days. A severe
one-off delinquency carries different meaning to chronic mild lateness.

## Affordability ratios
**ANNUITY_INCOME_RATIO** — Annual loan repayment divided by annual income. This is the
standard debt-service ratio. Values above 0.35 indicate strain.

**CREDIT_INCOME_RATIO** — Total credit requested divided by annual income. Values above
6.0 warrant closer review.

**CREDIT_TERM** — Annuity divided by total credit. Lower values imply a longer repayment
period and therefore longer exposure to changes in the borrower's circumstances.

**INCOME_PER_PERSON** — Household income divided by family members, a proxy for
disposable income.

## Credit history at other institutions
**BUREAU_ACTIVE_COUNT** — Number of currently active credit lines held elsewhere. High
counts indicate existing obligations.

**BUREAU_OVERDUE_COUNT** — Number of bureau-reported credit lines currently overdue. Any
value above zero is a material concern.

## Employment and stability
**YEARS_EMPLOYED** — Years in current employment. Short tenure correlates with income
instability.

**DAYS_EMPLOYED_ANOM** — Flag indicating the employment field contained a sentinel value
(365243), which denotes an applicant who is not employed, typically a pensioner.

## Prior relationship
**PREV_REFUSAL_RATE** — Proportion of the applicant's previous applications to this
institution that were refused.