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

| model        | condition    |   n_pairs |      mean |    median |   frac_pos |   mean_boot_ci95_low |   mean_boot_ci95_high |    cohend |
|:-------------|:-------------|----------:|----------:|----------:|-----------:|---------------------:|----------------------:|----------:|
| gemma3-1b-pt | plain_array  |        45 | nan       | nan       |   0        |            nan       |             nan       | nan       |
| gemma3-1b-pt | return_array |        45 | nan       | nan       |   0        |            nan       |             nan       | nan       |
| gemma3-1b-pt | subscribe    |        45 | nan       | nan       |   0        |            nan       |             nan       | nan       |
| gemma3-1b-pt | useEffect    |        45 | nan       | nan       |   0        |            nan       |             nan       | nan       |
| llama3.2-1b  | plain_array  |        45 |   1.42439 |   1.42969 |   1        |              1.29601 |               1.55859 |   3.20705 |
| llama3.2-1b  | return_array |        45 |   2.96319 |   2.96875 |   1        |              2.83759 |               3.08612 |   6.76402 |
| llama3.2-1b  | subscribe    |        45 |   1.19262 |   1.27734 |   1        |              1.07248 |               1.30565 |   3.0004  |
| llama3.2-1b  | useEffect    |        45 |   1.2842  |   1.28516 |   0.977778 |              1.14253 |               1.41823 |   2.66804 |

## Paired contrast vs. useEffect baseline

| model        | condition    |   n |   mean_diff_vs_baseline |         t |   p_twosided |     cohend |
|:-------------|:-------------|----:|------------------------:|----------:|-------------:|-----------:|
| gemma3-1b-pt | plain_array  |  45 |             nan         | nan       | nan          | nan        |
| gemma3-1b-pt | return_array |  45 |             nan         | nan       | nan          | nan        |
| gemma3-1b-pt | subscribe    |  45 |             nan         | nan       | nan          | nan        |
| gemma3-1b-pt | useEffect    |  45 |               0         | nan       | nan          | nan        |
| llama3.2-1b  | plain_array  |  45 |               0.140191  |   1.88768 |   0.0656746  |   0.281399 |
| llama3.2-1b  | return_array |  45 |               1.67899   |  24.0298  |   6.2272e-27 |   3.58215  |
| llama3.2-1b  | subscribe    |  45 |              -0.0915799 |  -2.58014 |   0.0132909  |  -0.384625 |
| llama3.2-1b  | useEffect    |  45 |               0         | nan       | nan          | nan        |

## Outputs

- `reactivity_ablationB_logits_3tokens.csv`
- `reactivity_ablationB_wide_per_sample.csv`
- `reactivity_ablationB_pairlevel_test.csv`
- `reactivity_ablationB_regression.csv`
