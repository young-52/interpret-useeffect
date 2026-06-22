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

| model        | condition   |   n_pairs |       mean |    median |   frac_pos |   mean_boot_ci95_low |   mean_boot_ci95_high |   t_p_onesided_greater |     cohend |
|:-------------|:------------|----------:|-----------:|----------:|-----------:|---------------------:|----------------------:|-----------------------:|-----------:|
| gemma3-1b-pt | subscribe   |        45 | nan        | nan       |   0        |           nan        |            nan        |          nan           | nan        |
| gemma3-1b-pt | useEffect   |        45 | nan        | nan       |   0        |           nan        |            nan        |          nan           | nan        |
| llama3.2-1b  | subscribe   |        45 |   0.372656 |   0.3125  |   0.844444 |             0.247743 |              0.496094 |            2.92778e-07 |   0.870129 |
| llama3.2-1b  | useEffect   |        45 |  -1.5947   |  -1.56641 |   0        |            -1.72917  |             -1.46527  |            1           |  -3.45037  |

## Stage A: all bodyuse cells, pair-level mean

| model        | condition   | bodyuse      |   n_pairs |       mean |     median |   frac_pos |   mean_boot_ci95_low |   mean_boot_ci95_high |     cohend |
|:-------------|:------------|:-------------|----------:|-----------:|-----------:|-----------:|---------------------:|----------------------:|-----------:|
| gemma3-1b-pt | subscribe   | both         |        45 | nan        | nan        |   0        |           nan        |            nan        | nan        |
| gemma3-1b-pt | subscribe   | control_only |        45 | nan        | nan        |   0        |           nan        |            nan        | nan        |
| gemma3-1b-pt | subscribe   | neither      |        45 | nan        | nan        |   0        |           nan        |            nan        | nan        |
| gemma3-1b-pt | subscribe   | target_only  |        45 | nan        | nan        |   0        |           nan        |            nan        | nan        |
| gemma3-1b-pt | useEffect   | both         |        45 | nan        | nan        |   0        |           nan        |            nan        | nan        |
| gemma3-1b-pt | useEffect   | control_only |        45 | nan        | nan        |   0        |           nan        |            nan        | nan        |
| gemma3-1b-pt | useEffect   | neither      |        45 | nan        | nan        |   0        |           nan        |            nan        | nan        |
| gemma3-1b-pt | useEffect   | target_only  |        45 | nan        | nan        |   0        |           nan        |            nan        | nan        |
| llama3.2-1b  | subscribe   | both         |        45 |   1.13655  |   1.10938  |   1        |             1.05213  |              1.22101  |   3.85724  |
| llama3.2-1b  | subscribe   | control_only |        45 |   0.372656 |   0.3125   |   0.844444 |             0.247743 |              0.496094 |   0.870129 |
| llama3.2-1b  | subscribe   | neither      |        45 |   0.625694 |   0.589844 |   0.933333 |             0.516484 |              0.735679 |   1.64995  |
| llama3.2-1b  | subscribe   | target_only  |        45 |   1.39566  |   1.32422  |   1        |             1.2809   |              1.51459  |   3.45397  |
| llama3.2-1b  | useEffect   | both         |        45 |   0.816146 |   0.732422 |   1        |             0.725434 |              0.90968  |   2.5787   |
| llama3.2-1b  | useEffect   | control_only |        45 |  -1.5947   |  -1.56641  |   0        |            -1.72917  |             -1.46527  |  -3.45037  |
| llama3.2-1b  | useEffect   | neither      |        45 |   0.876215 |   0.828125 |   0.977778 |             0.725343 |              1.02293  |   1.71461  |
| llama3.2-1b  | useEffect   | target_only  |        45 |   3.00981  |   2.95703  |   1        |             2.85946  |              3.16051  |   5.79438  |

## Ordering check: bodyuse vs. `both` (paired by pair_id)

| model        | condition   | bodyuse      |   n |   mean_diff_vs_baseline |         t |    p_twosided |     cohend |
|:-------------|:------------|:-------------|----:|------------------------:|----------:|--------------:|-----------:|
| gemma3-1b-pt | useEffect   | both         |  45 |               0         | nan       | nan           | nan        |
| gemma3-1b-pt | useEffect   | control_only |  45 |             nan         | nan       | nan           | nan        |
| gemma3-1b-pt | useEffect   | neither      |  45 |             nan         | nan       | nan           | nan        |
| gemma3-1b-pt | useEffect   | target_only  |  45 |             nan         | nan       | nan           | nan        |
| gemma3-1b-pt | subscribe   | both         |  45 |               0         | nan       | nan           | nan        |
| gemma3-1b-pt | subscribe   | control_only |  45 |             nan         | nan       | nan           | nan        |
| gemma3-1b-pt | subscribe   | neither      |  45 |             nan         | nan       | nan           | nan        |
| gemma3-1b-pt | subscribe   | target_only  |  45 |             nan         | nan       | nan           | nan        |
| llama3.2-1b  | useEffect   | both         |  45 |               0         | nan       | nan           | nan        |
| llama3.2-1b  | useEffect   | control_only |  45 |              -2.41085   | -41.5013  |   6.35663e-37 |  -6.18664  |
| llama3.2-1b  | useEffect   | neither      |  45 |               0.0600694 |   1.21767 |   0.229841    |   0.181519 |
| llama3.2-1b  | useEffect   | target_only  |  45 |               2.19366   |  35.8144  |   3.46684e-34 |   5.33889  |
| llama3.2-1b  | subscribe   | both         |  45 |               0         | nan       | nan           | nan        |
| llama3.2-1b  | subscribe   | control_only |  45 |              -0.763889  | -18.9533  |   8.62107e-23 |  -2.82539  |
| llama3.2-1b  | subscribe   | neither      |  45 |              -0.510851  | -14.3683  |   3.17587e-18 |  -2.1419   |
| llama3.2-1b  | subscribe   | target_only  |  45 |               0.259115  |   6.50178 |   6.16597e-08 |   0.969228 |

## Optional regression (bodyuse=both subset, decl_to_bracket distance as covariate)

| model        | term                        |   estimate |     std_err |     ci_low |    ci_high |       p_value |
|:-------------|:----------------------------|-----------:|------------:|-----------:|-----------:|--------------:|
| gemma3-1b-pt | role_dep_useState_form_code | nan        | nan         | nan        | nan        | nan           |
| gemma3-1b-pt | distance_c                  | nan        | nan         | nan        | nan        | nan           |
| llama3.2-1b  | role_dep_useState_form_code |   0.816146 |   0.0473783 |   0.723286 |   0.909006 |   1.69061e-66 |
| llama3.2-1b  | distance_c                  |   0        |   0         |   0        |   0        | nan           |

## Outputs

- `reactivity_ablationC_logits_3tokens.csv`
- `reactivity_ablationC_wide_per_sample.csv`
- `reactivity_ablationC_pairlevel_test.csv`
- `reactivity_ablationC_regression.csv`
