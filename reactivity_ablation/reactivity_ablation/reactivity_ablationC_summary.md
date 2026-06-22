# Reactivity Ablation C Summary (body-use)

## Dataset

- rows: 3600
- contexts: useEffect, subscribe

## Sign Convention

`D = logit(dep) - logit(alt)`. `LD_stateform = (role_dep_useState_form ? +1 : -1) * D`.

## Pre-registered predictions (Ablation C decision cell: control_only)

These were fixed before reporting any control_only numbers.

- `control_only` LD_stateform > 0 -> declaration prior beats body-use retrieval (pattern 1).
- `control_only` LD_stateform < 0 -> body-use retrieval dominates, state-form is only a
  tie-breaker (pattern 3).
- If declaration prior and body-use retrieval combine additively, the expected pair-level
  mean ordering is `target_only > both > control_only`.
- `neither` is auxiliary only (an empty body pushes both target/control logits low and
  noisy, with the array close token likely favored) -- no claim is based on it.
- Any cross-bodyuse magnitude comparison must be read alongside
  `decl_to_bracket_token_distance`, which differs by construction across bodyuse cells.

## Headline: control_only one-sample test vs 0 (decision cell)

| model        | condition   |   n_pairs |      mean |   median |   frac_pos |   mean_boot_ci95_low |   mean_boot_ci95_high |   t_p_onesided_greater |    cohend |
|:-------------|:------------|----------:|----------:|---------:|-----------:|---------------------:|----------------------:|-----------------------:|----------:|
| gemma3-1b-pt | subscribe   |        45 | -0.966667 | -0.90625 |  0.0444444 |            -1.12292  |             -0.809722 |            1           | -1.77929  |
| gemma3-1b-pt | useEffect   |        45 | -3.28229  | -3.1875  |  0         |            -3.45661  |             -3.10903  |            1           | -5.44844  |
| llama3.2-1b  | subscribe   |        45 |  0.352778 |  0.28125 |  0.822222  |             0.227083 |              0.477083 |            8.20022e-07 |  0.824562 |
| llama3.2-1b  | useEffect   |        45 | -1.59792  | -1.5625  |  0         |            -1.73472  |             -1.46597  |            1           | -3.40662  |

## Stage A: all bodyuse cells, pair-level mean

| model        | condition   | bodyuse      |   n_pairs |      mean |    median |   frac_pos |   mean_boot_ci95_low |   mean_boot_ci95_high |    cohend |
|:-------------|:------------|:-------------|----------:|----------:|----------:|-----------:|---------------------:|----------------------:|----------:|
| gemma3-1b-pt | subscribe   | both         |        45 |  0.746181 |  0.78125  |  1         |             0.653641 |              0.837674 |  2.35412  |
| gemma3-1b-pt | subscribe   | control_only |        45 | -0.966667 | -0.90625  |  0.0444444 |            -1.12292  |             -0.809722 | -1.77929  |
| gemma3-1b-pt | subscribe   | neither      |        45 |  1.19687  |  1.21875  |  1         |             1.08854  |              1.30418  |  3.24005  |
| gemma3-1b-pt | subscribe   | target_only  |        45 |  3.34271  |  3.25     |  1         |             3.1743   |              3.52535  |  5.47946  |
| gemma3-1b-pt | useEffect   | both         |        45 |  0.592535 |  0.578125 |  1         |             0.519444 |              0.66355  |  2.35684  |
| gemma3-1b-pt | useEffect   | control_only |        45 | -3.28229  | -3.1875   |  0         |            -3.45661  |             -3.10903  | -5.44844  |
| gemma3-1b-pt | useEffect   | neither      |        45 |  1.24514  |  1.23438  |  1         |             1.15243  |              1.34167  |  3.79006  |
| gemma3-1b-pt | useEffect   | target_only  |        45 |  6.16007  |  6.09375  |  1         |             5.91771  |              6.41111  |  7.19902  |
| llama3.2-1b  | subscribe   | both         |        45 |  1.12986  |  1.10938  |  1         |             1.04478  |              1.21458  |  3.84506  |
| llama3.2-1b  | subscribe   | control_only |        45 |  0.352778 |  0.28125  |  0.822222  |             0.227083 |              0.477083 |  0.824562 |
| llama3.2-1b  | subscribe   | neither      |        45 |  0.628472 |  0.59375  |  0.933333  |             0.520122 |              0.738889 |  1.6518   |
| llama3.2-1b  | subscribe   | target_only  |        45 |  1.40903  |  1.34375  |  1         |             1.29236  |              1.52849  |  3.43264  |
| llama3.2-1b  | useEffect   | both         |        45 |  0.823611 |  0.765625 |  1         |             0.734722 |              0.916667 |  2.6457   |
| llama3.2-1b  | useEffect   | control_only |        45 | -1.59792  | -1.5625   |  0         |            -1.73472  |             -1.46597  | -3.40662  |
| llama3.2-1b  | useEffect   | neither      |        45 |  0.875694 |  0.875    |  0.955556  |             0.726389 |              1.02224  |  1.71999  |
| llama3.2-1b  | useEffect   | target_only  |        45 |  3.01042  |  2.9375   |  1         |             2.86318  |              3.15903  |  5.90003  |

## Ordering check: bodyuse vs. `both` (paired by pair_id)

| model        | condition   | bodyuse      |   n |   mean_diff_vs_baseline |         t |    p_twosided |     cohend |
|:-------------|:------------|:-------------|----:|------------------------:|----------:|--------------:|-----------:|
| gemma3-1b-pt | useEffect   | both         |  45 |               0         | nan       | nan           | nan        |
| gemma3-1b-pt | useEffect   | control_only |  45 |              -3.87483   | -49.4876  |   3.23179e-40 |  -7.37717  |
| gemma3-1b-pt | useEffect   | neither      |  45 |               0.652604  |  18.2899  |   3.47816e-22 |   2.7265   |
| gemma3-1b-pt | useEffect   | target_only  |  45 |               5.56753   |  46.8795  |   3.35088e-39 |   6.98839  |
| gemma3-1b-pt | subscribe   | both         |  45 |               0         | nan       | nan           | nan        |
| gemma3-1b-pt | subscribe   | control_only |  45 |              -1.71285   | -31.1021  |   1.36522e-31 |  -4.63642  |
| gemma3-1b-pt | subscribe   | neither      |  45 |               0.450694  |  11.5991  |   5.67446e-15 |   1.7291   |
| gemma3-1b-pt | subscribe   | target_only  |  45 |               2.59653   |  30.0525  |   5.7855e-31  |   4.47996  |
| llama3.2-1b  | useEffect   | both         |  45 |               0         | nan       | nan           | nan        |
| llama3.2-1b  | useEffect   | control_only |  45 |              -2.42153   | -41.1554  |   9.10363e-37 |  -6.13509  |
| llama3.2-1b  | useEffect   | neither      |  45 |               0.0520833 |   1.03665 |   0.305564    |   0.154534 |
| llama3.2-1b  | useEffect   | target_only  |  45 |               2.18681   |  35.9649  |   2.89989e-34 |   5.36133  |
| llama3.2-1b  | subscribe   | both         |  45 |               0         | nan       | nan           | nan        |
| llama3.2-1b  | subscribe   | control_only |  45 |              -0.777083  | -19.3566  |   3.76142e-23 |  -2.88551  |
| llama3.2-1b  | subscribe   | neither      |  45 |              -0.501389  | -14.2604  |   4.17857e-18 |  -2.12581  |
| llama3.2-1b  | subscribe   | target_only  |  45 |               0.279167  |   6.77618 |   2.43515e-08 |   1.01013  |

## Optional regression (bodyuse=both subset, decl_to_bracket distance as covariate)

| model        | term                        |   estimate |   std_err |   ci_low |   ci_high |       p_value |
|:-------------|:----------------------------|-----------:|----------:|---------:|----------:|--------------:|
| gemma3-1b-pt | role_dep_useState_form_code |   0.592535 | 0.0376355 | 0.518771 |  0.666299 |   7.54471e-56 |
| gemma3-1b-pt | distance_c                  |   0        | 0         | 0        |  0        | nan           |
| llama3.2-1b  | role_dep_useState_form_code |   0.823611 | 0.0466009 | 0.732275 |  0.914947 |   6.68508e-70 |
| llama3.2-1b  | distance_c                  |   0        | 0         | 0        |  0        | nan           |

## Outputs

- `reactivity_ablationC_logits_3tokens.csv`
- `reactivity_ablationC_wide_per_sample.csv`
- `reactivity_ablationC_pairlevel_test.csv`
- `reactivity_ablationC_regression.csv`
