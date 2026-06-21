# Reactivity Pilot -- Statistical Inference Summary

범위: `reactivity_logits_3tokens.csv`(이미 측정된 logit)만 입력으로 쓰는 통계 분석이다. 추가 logit 측정이나 activation patching은 포함하지 않는다.

`reactive_var`/`stable_var`/`dep`/`alt`는 모두 useState/const 선언 형태에 붙인 라벨이며, reactivity 그 자체를 측정한 것이 아니다. 이 문서가 지지하는 가장 강한 주장은 "useState-form / declared-reactivity sensitivity"이며, "React reactivity 의미 이해" 수준의 주장은 하지 않는다.

## 0. 데이터 및 균형설계 확인

입력: `reactivity_logits_3tokens.csv`, 720 row (45 pair x 8 cell x 2 model). 모델별 모든 pair가 state_role/decl_order/body_order/reactive_body_pos 네 축 모두 4/4로 균형 잡힌 완전요인설계임을 확인했다.

## 1. Stage A -- pair-level 검정 (position 등 상쇄 후 useState-form 효과)

각 (model, pair)의 8 cell `LD_reactive` 평균(pair-mean)이 0보다 큰지 단측 검정했다. 이 평균은 state_role/decl_order/body_order 4/4 균형 덕분에 position·decl_order·identity bias가 상쇄된 값이다.

| model | n pairs | mean | median | sd | frac>0 | t (one-sided p) | Cohen's d | mean 95% CI | Wilcoxon (one-sided p) | rank-biserial r | median bootstrap 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 45 | 1.0713 | 1.0299 | 0.2438 | 1.000 | 29.4744 (6.536e-31) | 4.3938 | [0.9981, 1.1446] | 1035.0000 (2.842e-14) | 1.0000 | [0.9574, 1.1445] |
| Gemma 3 1B-pt | 45 | 0.3605 | 0.3768 | 0.3053 | 0.844 | 7.9206 (2.623e-10) | 1.1807 | [0.2688, 0.4522] | 976.0000 (2.579e-09) | 0.8860 | [0.2774, 0.4906] |

양측 p값(참고용): Llama 3.2 1B t=1.307e-30, Wilcoxon=5.684e-14; Gemma 3 1B-pt t=5.246e-10, Wilcoxon=5.157e-09

다중비교: 모델 2개 x 검정 2개(t-test, Wilcoxon) = 4개의 단측 검정을 했다. 보정하지 않은 raw p값을 그대로 보고했다. 참고용 Bonferroni 임계값(alpha=0.05): 0.01250.

### 두 검정/두 모델 일치 여부

| model | t-test p (one-sided) | Wilcoxon p (one-sided) | 둘 다 p<0.05 | 둘 다 p<Bonferroni |
|---|---|---|---|---|
| Llama 3.2 1B | 6.536e-31 | 2.842e-14 | True | True |
| Gemma 3 1B-pt | 2.623e-10 | 2.579e-09 | True | True |

## 2. Stage B -- 회귀 분해 (주효과 모델)

`D = logit(dep) - logit(alt)`를 `pos_dep_first`(position) + `role_dep_reactive`(useState-form) + `decl_dep_first`(declaration order)로 회귀했다. 세 predictor 모두 +-1 코딩이며, 균형 2x2x2 설계에서 서로 완전히 직교함을 확인했다(상호 상관 ~0).

| model | method | term | estimate | std err | 95% CI | p value |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | mixedlm | Intercept | 0.4397 | 0.0815 | [0.2799, 0.5995] | 6.959e-08 |
| Llama 3.2 1B | mixedlm | pos_dep_first | 1.2272 | 0.0243 | [1.1796, 1.2748] | 0.000e+00 |
| Llama 3.2 1B | mixedlm | role_dep_reactive | 1.0713 | 0.0243 | [1.0237, 1.1190] | 0.000e+00 |
| Llama 3.2 1B | mixedlm | decl_dep_first | 0.4114 | 0.0243 | [0.3638, 0.4590] | 2.494e-64 |
| Llama 3.2 1B | ols_cluster | Intercept | 0.4397 | 0.0819 | [0.2792, 0.6002] | 7.891e-08 |
| Llama 3.2 1B | ols_cluster | pos_dep_first | 1.2272 | 0.0377 | [1.1534, 1.3010] | 5.818e-233 |
| Llama 3.2 1B | ols_cluster | role_dep_reactive | 1.0713 | 0.0365 | [0.9998, 1.1429] | 2.321e-189 |
| Llama 3.2 1B | ols_cluster | decl_dep_first | 0.4114 | 0.0229 | [0.3665, 0.4563] | 4.448e-72 |
| Gemma 3 1B-pt | mixedlm | Intercept | 0.5894 | 0.0877 | [0.4175, 0.7613] | 1.800e-11 |
| Gemma 3 1B-pt | mixedlm | pos_dep_first | 1.1432 | 0.0265 | [1.0913, 1.1950] | 0.000e+00 |
| Gemma 3 1B-pt | mixedlm | role_dep_reactive | 0.3605 | 0.0265 | [0.3086, 0.4123] | 2.741e-42 |
| Gemma 3 1B-pt | mixedlm | decl_dep_first | 0.2002 | 0.0265 | [0.1484, 0.2521] | 3.765e-14 |
| Gemma 3 1B-pt | ols_cluster | Intercept | 0.5894 | 0.0881 | [0.4168, 0.7620] | 2.183e-11 |
| Gemma 3 1B-pt | ols_cluster | pos_dep_first | 1.1432 | 0.0354 | [1.0737, 1.2126] | 3.139e-228 |
| Gemma 3 1B-pt | ols_cluster | role_dep_reactive | 0.3605 | 0.0457 | [0.2709, 0.4501] | 3.084e-15 |
| Gemma 3 1B-pt | ols_cluster | decl_dep_first | 0.2002 | 0.0227 | [0.1557, 0.2448] | 1.256e-18 |

MixedLM 수렴 여부:
- Llama 3.2 1B (main_effects): converged=True, fit warning 0건
- Gemma 3 1B-pt (main_effects): converged=True, fit warning 0건

**해석 메모(자동 판정 아님):** `role_dep_reactive` 계수가 useState-form / declared-reactivity 효과의 크기다. `pos_dep_first`는 같은 단위의 순수 position(surface order) 효과이며, `role_dep_reactive`와 비교 가능하다 -- 어느 쪽이 더 크고 작은지는 위 표의 estimate와 CI를 보고 연구자가 판단한다. `decl_dep_first`는 선언 순서라는 2차 nuisance factor이고, 절편은 position/role/decl_order로 설명되지 않는 dep/alt identity 자체의 잔차 편향이다.

## 3. 손-추정치 교차 확인 (점검 신호, claim 아님)

Stage B 회귀계수가 사전에 손으로 어림한 값과 대체로 정합하는지 보는 점검 셀이다. 정합하지 않는 항목이 있다면 그 자체가 데이터나 도출 로직을 다시 볼 신호로 남겨둔다.

| model | term | hand estimate | MixedLM estimate | abs diff |
|---|---|---|---|---|
| Llama 3.2 1B | pos_dep_first | 1.20 | 1.2272 | 0.0272 |
| Llama 3.2 1B | role_dep_reactive | 1.10 | 1.0713 | 0.0287 |
| Gemma 3 1B-pt | pos_dep_first | 1.10 | 1.1432 | 0.0432 |
| Gemma 3 1B-pt | role_dep_reactive | 0.44 | 0.3605 | 0.0795 |

## 4. 확장 모델 -- position x useState-form 상호작용 (exploratory)

주효과 모델에 `pos_dep_first:role_dep_reactive` 상호작용항을 추가한 결과다. 메인 결과를 대체하지 않는 추가 진단이다.

| model | method | term | estimate | std err | 95% CI | p value |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | mixedlm | Intercept | 0.4397 | 0.0815 | [0.2799, 0.5995] | 6.959e-08 |
| Llama 3.2 1B | mixedlm | pos_dep_first | 1.2272 | 0.0243 | [1.1796, 1.2747] | 0.000e+00 |
| Llama 3.2 1B | mixedlm | role_dep_reactive | 1.0713 | 0.0243 | [1.0238, 1.1189] | 0.000e+00 |
| Llama 3.2 1B | mixedlm | decl_dep_first | 0.4114 | 0.0243 | [0.3639, 0.4590] | 1.521e-64 |
| Llama 3.2 1B | mixedlm | pos_dep_first:role_dep_reactive | -0.0349 | 0.0243 | [-0.0825, 0.0126] | 1.500e-01 |
| Llama 3.2 1B | ols_cluster | Intercept | 0.4397 | 0.0820 | [0.2790, 0.6004] | 8.228e-08 |
| Llama 3.2 1B | ols_cluster | pos_dep_first | 1.2272 | 0.0377 | [1.1533, 1.3011] | 2.589e-232 |
| Llama 3.2 1B | ols_cluster | role_dep_reactive | 1.0713 | 0.0366 | [0.9997, 1.1430] | 7.793e-189 |
| Llama 3.2 1B | ols_cluster | decl_dep_first | 0.4114 | 0.0229 | [0.3665, 0.4564] | 7.004e-72 |
| Llama 3.2 1B | ols_cluster | pos_x_role | -0.0349 | 0.0183 | [-0.0708, 0.0009] | 5.627e-02 |
| Gemma 3 1B-pt | mixedlm | Intercept | 0.5894 | 0.0877 | [0.4175, 0.7613] | 1.800e-11 |
| Gemma 3 1B-pt | mixedlm | pos_dep_first | 1.1432 | 0.0261 | [1.0919, 1.1944] | 0.000e+00 |
| Gemma 3 1B-pt | mixedlm | role_dep_reactive | 0.3605 | 0.0261 | [0.3092, 0.4117] | 3.041e-43 |
| Gemma 3 1B-pt | mixedlm | decl_dep_first | 0.2002 | 0.0261 | [0.1490, 0.2515] | 1.896e-14 |
| Gemma 3 1B-pt | mixedlm | pos_dep_first:role_dep_reactive | -0.0755 | 0.0261 | [-0.1268, -0.0243] | 3.858e-03 |
| Gemma 3 1B-pt | ols_cluster | Intercept | 0.5894 | 0.0882 | [0.4166, 0.7623] | 2.328e-11 |
| Gemma 3 1B-pt | ols_cluster | pos_dep_first | 1.1432 | 0.0355 | [1.0736, 1.2127] | 1.355e-227 |
| Gemma 3 1B-pt | ols_cluster | role_dep_reactive | 0.3605 | 0.0458 | [0.2708, 0.4502] | 3.370e-15 |
| Gemma 3 1B-pt | ols_cluster | decl_dep_first | 0.2002 | 0.0228 | [0.1556, 0.2448] | 1.403e-18 |
| Gemma 3 1B-pt | ols_cluster | pos_x_role | -0.0755 | 0.0190 | [-0.1127, -0.0384] | 6.817e-05 |

## 5. Limitation

1. `reactive_var`/`stable_var`/`dep`/`alt`는 useState/const 선언 형태에 붙인 라벨이며 reactivity 그 자체를 측정한 것이 아니다. 유의한 양의 `role_dep_reactive` 효과가 나와도 허용 claim은 "useState-form / declared-reactivity sensitivity"까지다.
2. useState 배정에는 setter(`setX`)와 destructuring(`const [X, setX] = ...`) 구조가 항상 동반된다. 이 design은 reactivity 자체와 setter/destructuring 형태를 분리하지 못한다.
3. Stage A/B 모두 `useEffect` 단일 condition, dep/alt 변수 pair 풀, prefix-only 입력이라는 동일한 데이터셋 범위 안에서만 추정한 것이다.
4. 1B 모델(Llama 3.2 1B, Gemma 3 1B-pt) 두 개에 대해서만 추정했다.
5. Stage A의 검정과 Stage B의 회귀는 같은 데이터를 다른 방식으로 본 것이라 서로 독립적인 증거가 아니다 -- 둘이 같은 방향을 보여주는 것은 일관성 확인이지 독립 재현이 아니다.
6. 다중비교(Stage A의 2 모델 x 2 검정)를 보정하지 않았다 -- raw p값을 그대로 제시했다.

## 6. 출력 파일

- `reactivity_pairlevel_test.csv`: Stage A 결과, 2 row.
- `reactivity_regression.csv`: Stage B 계수(주효과 + 상호작용, MixedLM + OLS), 36 row.
- `reactivity_inference_summary.md`: 본 문서.
