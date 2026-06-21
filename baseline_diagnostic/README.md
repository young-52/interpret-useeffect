# Baseline Diagnostic

측정 대상: 45 sample × 5 condition (`useEffect`, `useLayoutEffect`, `alias`, `alias_ctrl`, `subscribe`) × {clean, corrupted} × 2 model (Llama 3.2 1B, Gemma 3 1B-pt). 총 900개 (model, condition, sample, run_type) row.

입력은 모두 prefix-only이며 `[`로 끝난다. 측정 위치는 prefix 마지막 토큰(`[`) 위치이고, target token id는 각 prefix 문맥에서 직접 계산했다 (하드코딩 없음).

## 0. close_id 계산 방식 — §10.3 대비 변경점 (연구자 확인 완료)

§10.3 원안은 `prefix + "]"` 토큰화가 prefix 대비 정확히 1개 토큰만 늘어난다고 가정한다. 실제로는 prefix가 ` [`로 끝나기 때문에 `]`를 바로 이어붙이면 Llama 3.2 1B와 Gemma 3 1B-pt 토크나이저 모두 ` [` + `]`를 ` []`라는 **단일 병합 토큰**으로 재토큰화해서, 토큰 수가 늘지 않고 assert가 실패한다 (두 토크나이저 모두 `"[]"`가 단일 vocab 항목으로 존재).

대안: `prefix + dep + "]"` (예: `...}, [page]`)로 닫힌 1-원소 배열 문맥을 만들어 +2 토큰 assert로 검증하고, 마지막 토큰을 `close_id`로 취했다. 식별자가 대괄호 사이에 있을 때는 `]`가 항상 독립 토큰으로 분리되며 (`"[dep]"` → `['[', 'dep', ']']`), 이렇게 얻은 id가 standalone `"]"` 토큰화 결과와 정확히 일치함을 확인했다. 원안의 "하드코딩하지 말고 문맥에서 계산" 원칙은 유지하면서 빈 대괄호 병합 아티팩트만 회피한 것이다.

## 1. 전체 토큰 검증 (Stage 1)

45 sample × 5 condition × {clean, corrupted} = 450 조합에 대해 위 방식으로 dep_id/alt_id/close_id 계산 및 길이 assert를 수행했다.

| model | 통과 |
|---|---|
| Llama 3.2 1B | 450/450 |
| Gemma 3 1B-pt | 450/450 |

## 2. 조건별 요약 (전체 5조건)

`LD_copy = logit(dep) - logit(alt)`, `LD_inclusion = logit(dep) - logit(])`. 모든 값은 45 sample에 대한 median이다. `frac_gap_*_pos`는 `LD_*_clean > LD_*_corrupted`가 성립하는 sample 비율이다.

### Llama 3.2 1B

| condition | LD_inclusion clean | LD_inclusion corrupted | gap_inclusion | frac gap_inclusion>0 | LD_copy clean | LD_copy corrupted | gap_copy | frac gap_copy>0 | copy_error corrupted (median / mean) |
|---|---|---|---|---|---|---|---|---|---|
| useEffect | 14.117 | 10.853 | 3.479 | 1.000 | 4.110 | -1.883 | 6.305 | 1.000 | 12.696 / 12.631 |
| useLayoutEffect | 13.570 | 10.574 | 3.309 | 1.000 | 3.900 | -1.744 | 5.833 | 1.000 | 12.114 / 12.141 |
| alias | 13.567 | 11.332 | 2.267 | 1.000 | 2.954 | -1.438 | 4.607 | 1.000 | 12.723 / 12.714 |
| alias_ctrl | 13.309 | 11.679 | 1.623 | 1.000 | 2.503 | -0.457 | 3.195 | 1.000 | 12.322 / 12.327 |
| subscribe | 13.234 | 12.189 | 1.186 | 0.978 | 2.075 | 0.248 | 1.815 | 0.978 | 12.092 / 12.053 |

### Gemma 3 1B-pt

| condition | LD_inclusion clean | LD_inclusion corrupted | gap_inclusion | frac gap_inclusion>0 | LD_copy clean | LD_copy corrupted | gap_copy | frac gap_copy>0 | copy_error corrupted (median / mean) |
|---|---|---|---|---|---|---|---|---|---|
| useEffect | 14.563 | 9.802 | 4.560 | 1.000 | 7.564 | -4.384 | 11.929 | 1.000 | 14.062 / 14.177 |
| useLayoutEffect | 14.360 | 10.466 | 3.950 | 1.000 | 6.142 | -3.772 | 10.085 | 1.000 | 14.004 / 14.066 |
| alias | 13.079 | 10.699 | 2.041 | 1.000 | 3.703 | -1.904 | 5.667 | 1.000 | 12.765 / 12.877 |
| alias_ctrl | 13.024 | 11.701 | 1.182 | 1.000 | 2.897 | -0.917 | 3.752 | 1.000 | 12.632 / 12.672 |
| subscribe | 13.449 | 11.815 | 1.584 | 1.000 | 3.307 | -1.251 | 4.531 | 1.000 | 12.950 / 12.955 |

## 3. §12.1 기준 체크리스트 — `useEffect` 조건 중심

아래는 §12.1에서 `LD_inclusion = logit(dep) - logit(])`을 메인 metric으로 채택하기 위한 5개 기준을 `useEffect` 조건(n=45)에 대해 그대로 평가한 표다. 기준 1~4는 §12.1에 명시된 부등식이므로 통과 여부를 그대로 표시했다. 기준 5(`copy_error`가 과하게 양수로 치우치지 않음)는 고정 임계값이 문서에 없으므로 **수치만 제시**하고 판정은 하지 않았다. 메인 metric 최종 선택은 연구자가 이 표를 보고 결정한다.

| # | 기준 | Llama 3.2 1B | 판정 | Gemma 3 1B-pt | 판정 |
|---|---|---|---|---|---|
| 1 | median LD_inclusion_clean > 0 | 14.117 | PASS | 14.563 | PASS |
| 2 | median LD_inclusion_corrupted < 0 | 10.853 | FAIL | 9.802 | FAIL |
| 3 | median gap_inclusion > 0 | 3.479 | PASS | 4.560 | PASS |
| 4 | 샘플 다수에서 LD_clean > LD_corrupted | 1.000 | PASS | 1.000 | PASS |
| 5 | copy_error_corrupted가 과하게 양수로 치우치지 않음 (임계값 미정) | 12.696 | (판정 보류) | 14.062 | (판정 보류) |

### 기준 2 보강: sample 단위 분포 (`useEffect`, n=45)

median이 아니라 sample 전체에서 LD_inclusion_corrupted < 0가 성립하는 비율도 함께 보고한다 (median만으로는 분포의 일관성을 알 수 없기 때문).

| model | LD_inclusion_corrupted < 0 인 sample 비율 | min | max |
|---|---|---|---|
| Llama 3.2 1B | 0.000 | 8.675 | 12.153 |
| Gemma 3 1B-pt | 0.000 | 3.578 | 12.418 |

## 4. 참고: LD_copy(variable tracking) 분포 — `useEffect`, n=45

§12.2 판단에 참고할 수 있도록 동일한 방식으로 LD_copy 쪽 sample-level 분포도 함께 보고한다.

| model | LD_copy_corrupted < 0 인 sample 비율 | min | max |
|---|---|---|---|
| Llama 3.2 1B | 0.956 | -3.795 | 0.400 |
| Gemma 3 1B-pt | 1.000 | -9.643 | -2.410 |

## 5. 출력 파일

- `baseline_logits_3tokens.csv`: raw long format (model, condition, sample_id, run_type 별 logit(dep)/logit(alt)/logit(]) 및 파생 지표). 900 row.
- `baseline_wide_per_sample.csv`: 위 raw를 (model, condition, sample_id) 단위로 clean/corrupted를 나란히 두고 gap_copy/gap_inclusion을 계산한 중간 산출물.
- `baseline_summary_table.csv`: 본 문서 §2 표의 원본 수치.

## 6. 메인 metric 선택은 보류

본 문서는 §12 기준의 통과 여부와 수치만 제시한다. `LD_inclusion`을 메인으로 할지 `LD_copy`를 메인으로 할지, 혹은 §12.3의 mixed-result 처리를 적용할지는 연구자가 위 결과를 보고 결정한다.
