# Consumer lending credit policy (illustrative)

> This is a synthetic policy document created for demonstration purposes. It is modelled
> on standard consumer lending practice but does not represent any real institution.

## 1. Risk bands

Applications are assigned a band from the model's predicted probability of default:

- **Low risk** — PD below 0.08. Eligible for automated approval.
- **Medium risk** — PD from 0.08 to 0.15. Requires analyst review.
- **High risk** — PD above 0.15. Requires senior underwriter review; decline is the
  default outcome absent compensating factors.

## 2. Automated decisions

No application may be declined by automated process alone. The model produces a
**recommendation**; a human underwriter records the decision. All model-assisted
decisions must carry a recorded rationale.

## 3. Affordability thresholds

- Debt-service ratio (ANNUITY_INCOME_RATIO) above 0.35 requires documented justification.
- Credit-to-income ratio above 6.0 requires senior review regardless of model band.
- Applications where income cannot be verified are ineligible for automated approval.

## 4. Adverse credit triggers

Any of the following requires escalation to senior review:

- One or more overdue accounts at other institutions (BUREAU_OVERDUE_COUNT > 0)
- Late payment rate above 0.30 on prior instalments
- A maximum historical payment delay exceeding 90 days
- Three or more previous applications refused

## 5. Thin-file applicants

Where two or more external bureau scores are missing, the application is treated as a
thin file and may not be auto-approved irrespective of model output. Manual verification
of income and identity is required.

## 6. Prohibited factors

Decisions must not be made on the basis of protected characteristics. Where a model
feature acts as a proxy for a protected characteristic, its use must be reviewed by the
model risk function. Age may be used only where actuarially justified and permitted by
local regulation.

## 7. Explanation requirement

Every declined application must be accompanied by the principal reasons for the decision,
expressed in language the applicant can understand, drawn from the approved adverse-action
reason code list.

## 8. Model governance

The model is revalidated quarterly. Population stability is monitored monthly; a PSI above
0.25 on any top-ten feature triggers investigation and possible retraining.