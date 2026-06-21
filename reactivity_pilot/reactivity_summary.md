# Reactivity Pilot -- Logit Measurement Summary

범위: reactivity pilot의 logit 측정 단계(`sync_reactivity_pilot_resynthesis_v0_2.md` §10-§11)만 다룬다. activation patching은 포함하지 않는다. baseline diagnostic (`baseline_diagnostic/`)과는 다른 데이터셋ㆍ다른 metric을 쓰는 별도 실험이다.

`reactive_var`/`stable_var`는 useState/const 선언 형태에 붙인 라벨이며, reactivity 그 자체를 측정한 것이 아니다. 이 문서가 지지하는 가장 강한 주장은 "useState-form / declared-reactivity sensitivity"이며, "React reactivity 의미 이해" 수준의 주장은 하지 않는다.

## 0. 데이터셋 및 제외 내역 (Stage 1)

원본 데이터셋: 45 pair x 8 cell = 360 row. 두 모델(Llama 3.2 1B, Gemma 3 1B-pt) 모두에서 row 단위로 reactive_var/stable_var single-token 여부와 reactive_id != stable_id를 확인했다.

- Llama 3.2 1B: 360/360 row 통과
- Gemma 3 1B-pt: 360/360 row 통과

두 모델 모두 전체 row에서 통과했고, 제외된 pair는 없다.

이후 분석은 모두 `dataset_valid` (45 pair x 8 cell = 360 row) 기준이다.

## 1. 전체 결과 (Stage 3 -- overall)

`LD_reactive = logit(reactive_var) - logit(stable_var)`. 양수면 같은 prefix 안에서 모델이 const 형태보다 useState 형태 식별자에 더 높은 logit을 두었다는 뜻이다.

| model | n | median LD_reactive | mean LD_reactive | frac(LD_reactive>0) |
|---|---|---|---|---|
| Gemma 3 1B-pt | 360 | 0.438 | 0.360 | 0.603 |
| Llama 3.2 1B | 360 | 1.092 | 1.071 | 0.728 |

## 2. Factor별 분해 (marginal)

각 factor를 단독으로 나눈 marginal 집계다(다른 두 factor는 평균으로 깔려 있음). 완전교차 표는 §4와 `reactivity_summary_table.csv`에 별도로 있다.

### state_role

| model | state_role | n | median LD_reactive | mean LD_reactive | frac(LD_reactive>0) |
|---|---|---|---|---|---|
| Gemma 3 1B-pt | alt_reactive | 180 | -0.118 | -0.229 | 0.472 |
| Gemma 3 1B-pt | dep_reactive | 180 | 0.908 | 0.950 | 0.733 |
| Llama 3.2 1B | alt_reactive | 180 | 0.577 | 0.632 | 0.611 |
| Llama 3.2 1B | dep_reactive | 180 | 1.524 | 1.511 | 0.844 |

### decl_order

| model | decl_order | n | median LD_reactive | mean LD_reactive | frac(LD_reactive>0) |
|---|---|---|---|---|---|
| Gemma 3 1B-pt | reactive_first | 180 | 0.649 | 0.561 | 0.639 |
| Gemma 3 1B-pt | stable_first | 180 | 0.218 | 0.160 | 0.567 |
| Llama 3.2 1B | reactive_first | 180 | 1.479 | 1.483 | 0.817 |
| Llama 3.2 1B | stable_first | 180 | 0.854 | 0.660 | 0.639 |

### body_order

| model | body_order | n | median LD_reactive | mean LD_reactive | frac(LD_reactive>0) |
|---|---|---|---|---|---|
| Gemma 3 1B-pt | alt_first | 180 | 0.447 | 0.436 | 0.683 |
| Gemma 3 1B-pt | dep_first | 180 | 0.306 | 0.285 | 0.522 |
| Llama 3.2 1B | alt_first | 180 | 1.092 | 1.106 | 0.844 |
| Llama 3.2 1B | dep_first | 180 | 1.079 | 1.036 | 0.611 |

### reactive_body_pos (order-bias 통제 핵심)

| model | reactive_body_pos | n | median LD_reactive | mean LD_reactive | frac(LD_reactive>0) |
|---|---|---|---|---|---|
| Gemma 3 1B-pt | first | 180 | 1.524 | 1.504 | 0.950 |
| Gemma 3 1B-pt | second | 180 | -0.649 | -0.783 | 0.256 |
| Llama 3.2 1B | first | 180 | 2.317 | 2.299 | 1.000 |
| Llama 3.2 1B | second | 180 | -0.147 | -0.156 | 0.456 |

## 3. order effect 대 role effect 자동 비교 (초안)

`range_*`는 해당 factor의 두 레벨 사이 median LD_reactive 차이(max - min)다. `order_range_exceeds_role_range`가 True면 body_order/reactive_body_pos의 효과가 state_role 효과보다 크다는 뜻이고, 이는 sync note §9.2의 order-dominated negative 해석에 더 가깝다는 신호다. **이 표는 자동 비교 초안이며 최종 해석은 연구자가 한다.**

| model | range state_role | range decl_order | range body_order | range reactive_body_pos | order range > role range | sign consistent (8 combo) |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | 0.947 | 0.624 | 0.013 | 2.464 | True | False |
| Gemma 3 1B-pt | 1.026 | 0.430 | 0.140 | 2.173 | True | False |

## 4. 완전교차 표 (`model x state_role x decl_order x body_order x reactive_body_pos`)

원본은 `reactivity_summary_table.csv`. 모델당 최대 8개 조합(reactive_body_pos는 state_role과 body_order로 결정되므로 16이 아니라 8).

| model | state_role | decl_order | body_order | reactive_body_pos | n | median LD_reactive | mean LD_reactive | frac(LD_reactive>0) |
|---|---|---|---|---|---|---|---|---|
| Gemma 3 1B-pt | alt_reactive | reactive_first | alt_first | first | 45 | 1.395 | 1.337 | 0.933 |
| Gemma 3 1B-pt | alt_reactive | reactive_first | dep_first | second | 45 | -1.405 | -1.251 | 0.067 |
| Gemma 3 1B-pt | alt_reactive | stable_first | alt_first | first | 45 | 0.754 | 0.643 | 0.867 |
| Gemma 3 1B-pt | alt_reactive | stable_first | dep_first | second | 45 | -1.697 | -1.645 | 0.022 |
| Gemma 3 1B-pt | dep_reactive | reactive_first | alt_first | second | 45 | 0.114 | 0.027 | 0.556 |
| Gemma 3 1B-pt | dep_reactive | reactive_first | dep_first | first | 45 | 2.159 | 2.130 | 1.000 |
| Gemma 3 1B-pt | dep_reactive | stable_first | alt_first | second | 45 | -0.166 | -0.262 | 0.378 |
| Gemma 3 1B-pt | dep_reactive | stable_first | dep_first | first | 45 | 1.803 | 1.905 | 1.000 |
| Llama 3.2 1B | alt_reactive | reactive_first | alt_first | first | 45 | 2.458 | 2.424 | 1.000 |
| Llama 3.2 1B | alt_reactive | reactive_first | dep_first | second | 45 | -0.349 | -0.203 | 0.378 |
| Llama 3.2 1B | alt_reactive | stable_first | alt_first | first | 45 | 1.361 | 1.364 | 1.000 |
| Llama 3.2 1B | alt_reactive | stable_first | dep_first | second | 45 | -1.239 | -1.058 | 0.067 |
| Llama 3.2 1B | dep_reactive | reactive_first | alt_first | second | 45 | 0.605 | 0.642 | 0.889 |
| Llama 3.2 1B | dep_reactive | reactive_first | dep_first | first | 45 | 2.940 | 3.068 | 1.000 |
| Llama 3.2 1B | dep_reactive | stable_first | alt_first | second | 45 | -0.033 | -0.004 | 0.489 |
| Llama 3.2 1B | dep_reactive | stable_first | dep_first | first | 45 | 2.261 | 2.339 | 1.000 |

## 5. Auxiliary: close(`]`) 진단 (main 해석에 사용하지 않음)

| model | n_total | close 계산 성공 | close 계산 실패 | median LD_reactive_vs_close | median LD_stable_vs_close | frac(reactive < close) | frac(stable < close) |
|---|---|---|---|---|---|---|---|
| Gemma 3 1B-pt | 360 | 360 | 0 | 12.758 | 12.356 | 0.000 | 0.000 |
| Llama 3.2 1B | 360 | 360 | 0 | 13.468 | 12.468 | 0.000 | 0.000 |

## 6. 해석 가이드 (sync note §9, 자동 판정 아님)

- **positive**: §1의 median/frac이 두 모델 모두 양수 방향이고, §2의 세 factor(state_role / decl_order / body_order) swap에도 부호가 유지되며, §3에서 order range가 role range를 넘지 않는 경우. 이 경우 허용 claim은 "useState-form / declared-reactivity sensitivity가 관찰된다"까지이며, setterㆍdestructuring confound 한계(§7)를 함께 명시해야 한다.
- **order-dominated negative**: §3에서 body_order 또는 reactive_body_pos의 range가 state_role의 range보다 크면, 모델 예측이 declared-reactivity보다 surface order/copy salience에 더 가깝다는 신호다.
- **null/mixed**: §1의 median이 0 근처이거나 모델 간 부호가 불안정하면, 이번 operational contrast에서 안정적 증거가 없다는 뜻이다.
- 위 세 갈래 중 어디에 해당하는지, 그리고 최종 메인 claim의 표현은 이 표들을 검토한 연구자가 결정한다. 이 노트북은 판정하지 않는다.

## 7. Limitation

1. `reactive_var`/`stable_var`는 useState/const 선언 형태에 붙인 라벨이며 reactivity 그 자체를 측정한 것이 아니다.
2. positive 결과가 나와도, state_role swap을 하더라도 useState 배정에는 setter(`setX`)와 destructuring(`const [X, setX] = ...`) 구조가 항상 동반된다. 따라서 positive 결과는 "useState 선언 형태에 대한 민감성"까지만 지지하며, reactivity 자체 때문인지 setter/destructuring 형태 때문인지는 이 design으로 분리되지 않는다.
3. `close_target`(`]`) 관련 auxiliary 지표는 tokenization artifact의 영향을 받을 수 있어 main 결론에 쓰지 않는다.
4. 이 pilot은 `useEffect` 단일 condition만 다룬다. 다른 syntactic 환경(useLayoutEffect, alias, non-hook callback-array)으로의 일반화는 baseline diagnostic의 범위이고, 이 pilot에서는 다루지 않는다.
5. activation patching은 이 단계에 포함하지 않는다 -- 본 문서는 logit 측정과 집계까지다.
6. 1B 모델(Llama 3.2 1B, Gemma 3 1B-pt) 두 개에 대해서만 측정했다.

## 8. 출력 파일

- `reactivity_logits_3tokens.csv`: long raw, 720 row.
- `reactivity_wide_per_sample.csv`: pair 단위 wide + factor effect, 90 row.
- `reactivity_summary_table.csv`: 완전교차 집계, 16 row.
- `reactivity_summary.md`: 본 문서.
