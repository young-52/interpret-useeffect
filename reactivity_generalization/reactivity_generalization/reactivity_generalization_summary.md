# Reactivity Generalization Summary

## Dataset

- rows: 1800
- pairs: 45
- conditions: useEffect, useLayoutEffect, alias, alias_ctrl, subscribe
- structural validation: passed
- token validation: passed
- token-validation failures: 0
- failed pairs dropped/flagged: 0

## Sign Convention

`D = logit(dep) - logit(alt)`.

`LD_stateform = (role_dep_useState_form ? +1 : -1) * D`.

`LD_reactive` is emitted only as a backward-compatible alias for `LD_stateform`.

## Stage A

Pair-level mean of the eight cells per model/condition.

| model        | condition       |   n_pairs |   pair_mean_mean |   pair_mean_median |   frac_pair_mean_pos |   mean_boot_ci95_low |   mean_boot_ci95_high |   cohend |
|:-------------|:----------------|----------:|-----------------:|-------------------:|---------------------:|---------------------:|----------------------:|---------:|
| gemma3-1b-pt | alias           |        45 |         0.401736 |           0.4375   |             0.911111 |             0.322222 |              0.477778 | 1.49585  |
| gemma3-1b-pt | alias_ctrl      |        45 |         0.640972 |           0.640625 |             0.977778 |             0.540616 |              0.735764 | 1.90297  |
| gemma3-1b-pt | subscribe       |        45 |         0.750347 |           0.75     |             0.977778 |             0.640278 |              0.856597 | 2.00199  |
| gemma3-1b-pt | useEffect       |        45 |         0.353819 |           0.375    |             0.844444 |             0.2625   |              0.443056 | 1.14185  |
| gemma3-1b-pt | useLayoutEffect |        45 |         0.274306 |           0.328125 |             0.8      |             0.194792 |              0.354514 | 0.987751 |
| llama3.2-1b  | alias           |        45 |         0.944097 |           0.875    |             1        |             0.883333 |              1.00451  | 4.49879  |
| llama3.2-1b  | alias_ctrl      |        45 |         1.25069  |           1.20312  |             1        |             1.1875   |              1.3191   | 5.46769  |
| llama3.2-1b  | subscribe       |        45 |         1.31771  |           1.29688  |             1        |             1.24549  |              1.39132  | 5.1664   |
| llama3.2-1b  | useEffect       |        45 |         1.07569  |           1        |             1        |             1.00521  |              1.14897  | 4.37124  |
| llama3.2-1b  | useLayoutEffect |        45 |         0.993403 |           0.96875  |             1        |             0.923611 |              1.06806  | 3.97697  |

## Stage B

Regression dependent variable: `D = logit(dep) - logit(alt)`.

| model        | term                                     |   estimate |    std_err |       ci_low |    ci_high |      p_value |
|:-------------|:-----------------------------------------|-----------:|-----------:|-------------:|-----------:|-------------:|
| gemma3-1b-pt | role_dep_useState_form_code              |  0.353819  | 0.0463336  |  0.263007    |  0.444632  | 2.23483e-14  |
| gemma3-1b-pt | role_dep_useState_form_x_useLayoutEffect | -0.0795139 | 0.0168761  | -0.11259     | -0.0464374 | 2.4574e-06   |
| gemma3-1b-pt | role_dep_useState_form_x_alias           |  0.0479167 | 0.0242723  |  0.000343804 |  0.0954895 | 0.0483671    |
| gemma3-1b-pt | role_dep_useState_form_x_alias_ctrl      |  0.287153  | 0.0246471  |  0.238845    |  0.33546   | 2.2791e-31   |
| gemma3-1b-pt | role_dep_useState_form_x_subscribe       |  0.396528  | 0.0229259  |  0.351594    |  0.441462  | 5.03824e-67  |
| llama3.2-1b  | role_dep_useState_form_code              |  1.07569   | 0.0367967  |  1.00357     |  1.14781   | 7.29817e-188 |
| llama3.2-1b  | role_dep_useState_form_x_useLayoutEffect | -0.0822917 | 0.00799169 | -0.0979551   | -0.0666282 | 7.25745e-25  |
| llama3.2-1b  | role_dep_useState_form_x_alias           | -0.131597  | 0.0229972  | -0.176671    | -0.0865235 | 1.05089e-08  |
| llama3.2-1b  | role_dep_useState_form_x_alias_ctrl      |  0.175     | 0.0283736  |  0.119389    |  0.230611  | 6.92906e-10  |
| llama3.2-1b  | role_dep_useState_form_x_subscribe       |  0.242014  | 0.0308538  |  0.181542    |  0.302486  | 4.36798e-15  |

## Outputs

- `reactivity_generalization_logits_3tokens.csv`
- `reactivity_generalization_wide_per_sample.csv`
- `reactivity_generalization_pairlevel_test.csv`
- `reactivity_generalization_regression.csv`
