# E2 Model Structure

Importances are computed over the fitted transformed feature matrix. They describe this fitted model, not causal effects.

## Feature importance

|feature|importance|
|---|---|
|acm_award|0.934768|
|work_years|0.065232|

## Split nodes

|estimator|feature|threshold|samples|
|---|---|---|---|
|1|acm_award|0.5|20|
|2|acm_award|0.5|20|
|3|acm_award|0.5|20|
|4|acm_award|0.5|20|
|5|acm_award|0.5|20|
|6|acm_award|0.5|20|
|7|acm_award|0.5|20|
|8|acm_award|0.5|20|
|9|acm_award|0.5|20|
|10|acm_award|0.5|20|
|11|acm_award|0.5|20|
|12|acm_award|0.5|20|
|13|acm_award|0.5|20|
|14|acm_award|0.5|20|
|15|acm_award|0.5|20|
|16|acm_award|0.5|20|
|17|acm_award|0.5|20|
|18|acm_award|0.5|20|
|19|acm_award|0.5|20|
|20|acm_award|0.5|20|
|21|acm_award|0.5|20|
|22|work_years|9|20|
|23|acm_award|0.5|20|
|24|work_years|9|20|
|25|acm_award|0.5|20|
|26|work_years|9|20|
|27|acm_award|0.5|20|
|28|work_years|9|20|
|29|acm_award|0.5|20|
|30|work_years|9|20|
|31|acm_award|0.5|20|
|32|work_years|9|20|
|33|acm_award|0.5|20|
|34|acm_award|0.5|20|
|35|work_years|9|20|
|36|acm_award|0.5|20|
|37|work_years|9|20|
|38|acm_award|0.5|20|
|39|work_years|9|20|
|40|acm_award|0.5|20|

## P scores by ACM award indicator

|acm_award|count|score_min|score_max|predicted_positive|
|---|---|---|---|---|
|0|121|0.198877|0.325353|0|
|1|139|0.872024|0.929761|139|

P produced 4 distinct uncalibrated score tiers; equal scores remain tied.
