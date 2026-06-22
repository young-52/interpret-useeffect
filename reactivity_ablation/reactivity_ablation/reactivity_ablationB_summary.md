# Reactivity Ablation B Summary (context-shape, optional)

## Dataset

- rows: 1440
- contexts: useEffect, subscribe, plain_array, return_array

## Scope note (Ablation B, optional)

Headline framing is not fixed in advance. If the effect holds in `useEffect` and
`subscribe` but drops in `plain_array`/`return_array`, that reads as gated by
callback-array context. If it holds across all four, that reads as a broader
array-completion prior not specific to callback wrapping. The phrase "beyond dependency
arrays" is licensed only by the second pattern. Until resolved, all claims here are scoped
to callback-array completion, not dependency-array completion specifically.

## Stage A: pair-level mean per context

| model        | condition    |   n_pairs |     mean |   median |   frac_pos |   mean_boot_ci95_low |   mean_boot_ci95_high |   cohend |
|:-------------|:-------------|----------:|---------:|---------:|-----------:|---------------------:|----------------------:|---------:|
| gemma3-1b-pt | plain_array  |        45 | 1.1125   | 1.09375  |   1        |             0.995833 |              1.23195  |  2.72743 |
| gemma3-1b-pt | return_array |        45 | 0.748958 | 0.765625 |   1        |             0.6625   |              0.835764 |  2.49848 |
| gemma3-1b-pt | subscribe    |        45 | 2.22917  | 2.1875   |   1        |             2.11492  |              2.34444  |  5.54052 |
| gemma3-1b-pt | useEffect    |        45 | 2.35451  | 2.34375  |   1        |             2.24271  |              2.47188  |  5.99089 |
| llama3.2-1b  | plain_array  |        45 | 1.42778  | 1.40625  |   1        |             1.29722  |              1.5625   |  3.20357 |
| llama3.2-1b  | return_array |        45 | 2.96597  | 3        |   1        |             2.84026  |              3.09097  |  6.77558 |
| llama3.2-1b  | subscribe    |        45 | 1.19097  | 1.21875  |   1        |             1.0729   |              1.30208  |  3.03536 |
| llama3.2-1b  | useEffect    |        45 | 1.29028  | 1.28125  |   0.977778 |             1.15207  |              1.42014  |  2.75996 |

## Paired contrast vs. useEffect baseline

| model        | condition    |   n |   mean_diff_vs_baseline |         t |    p_twosided |     cohend |
|:-------------|:-------------|----:|------------------------:|----------:|--------------:|-----------:|
| gemma3-1b-pt | plain_array  |  45 |              -1.24201   | -16.8062  |   9.11458e-21 |  -2.50532  |
| gemma3-1b-pt | return_array |  45 |              -1.60556   | -28.6621  |   4.21174e-30 |  -4.2727   |
| gemma3-1b-pt | subscribe    |  45 |              -0.125347  |  -3.41146 |   0.0013957   |  -0.508551 |
| gemma3-1b-pt | useEffect    |  45 |               0         | nan       | nan           | nan        |
| llama3.2-1b  | plain_array  |  45 |               0.1375    |   1.86348 |   0.0690774   |   0.277791 |
| llama3.2-1b  | return_array |  45 |               1.67569   |  23.9702  |   6.89354e-27 |   3.57327  |
| llama3.2-1b  | subscribe    |  45 |              -0.0993056 |  -2.78139 |   0.0079394   |  -0.414625 |
| llama3.2-1b  | useEffect    |  45 |               0         | nan       | nan           | nan        |

## Outputs

- `reactivity_ablationB_logits_3tokens.csv`
- `reactivity_ablationB_wide_per_sample.csv`
- `reactivity_ablationB_pairlevel_test.csv`
- `reactivity_ablationB_regression.csv`
