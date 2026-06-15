"""
variable_names.json 토큰 검증 스크립트
각 항목의 모든 필드 값이 Llama 3.2 / Gemma 3 양쪽에서 단일 토큰인지 검증합니다.

Usage:
    python validate_tokens.py --json variable_names.json

Requirements:
    pip install transformers
    huggingface-cli login  (Llama는 gated model)
"""

import json
import argparse
from transformers import AutoTokenizer

LLAMA_MODEL = "meta-llama/Llama-3.2-1B"
GEMMA_MODEL = "google/gemma-3-1b-pt"

# clean_target / corrupted_target으로 쓰일 필드
# Component는 측정 대상이 아니므로 제외
FIELDS_TO_CHECK = ["dep", "alt", "init", "param"]


def is_single_token(tokenizer, word):
    """
    공백 포함/미포함 두 버전 모두 확인.
    단일 토큰이면 (True, 실제 사용할 형태) 반환.
    """
    with_space = tokenizer.encode(" " + word, add_special_tokens=False)
    without_space = tokenizer.encode(word, add_special_tokens=False)

    if len(with_space) == 1:
        return True, " " + word
    if len(without_space) == 1:
        return True, word
    return False, None


def validate(json_path, llama_tok, gemma_tok):
    with open(json_path) as f:
        items = json.load(f)

    passed_items = []
    failed_items = []

    print(f"\n{'=' * 70}")
    print(f"검증 대상 필드: {FIELDS_TO_CHECK}")
    print(f"총 {len(items)}개 항목")
    print(f"{'=' * 70}\n")

    for idx, item in enumerate(items):
        item_ok = True
        field_results = {}

        for field in FIELDS_TO_CHECK:
            value = item.get(field)
            if value is None:
                continue

            llama_ok, llama_form = is_single_token(llama_tok, value)
            gemma_ok, gemma_form = is_single_token(gemma_tok, value)
            passes = llama_ok and gemma_ok

            field_results[field] = {
                "value": value,
                "llama_ok": llama_ok,
                "llama_form": llama_form,
                "gemma_ok": gemma_ok,
                "gemma_form": gemma_form,
                "passes": passes,
            }

            if not passes:
                item_ok = False

        status = "✓" if item_ok else "✗"
        component = item.get("Component", f"item_{idx}")
        print(f"  {status} [{idx:02d}] Component={component}")

        for field, r in field_results.items():
            mark = "✓" if r["passes"] else "✗"
            detail = ""
            if not r["llama_ok"]:
                detail += f" Llama:✗"
            if not r["gemma_ok"]:
                detail += f" Gemma:✗"
            if r["llama_form"] != r["gemma_form"] and r["passes"]:
                detail += f" ⚠ form differs (Llama:'{r['llama_form']}' Gemma:'{r['gemma_form']}')"
            print(f"       {mark} {field}='{r['value']}'{detail}")

        if item_ok:
            # use_form: 두 토크나이저에서 공통으로 쓸 수 있는 형태
            # 공백 포함이 우선 (TransformerLens 기본)
            item_with_forms = dict(item)
            item_with_forms["_token_forms"] = {
                f: r["llama_form"] for f, r in field_results.items()
            }
            passed_items.append(item_with_forms)
        else:
            failed_items.append(item)

    # 결과 요약
    print(f"\n{'=' * 70}")
    print(f"결과 요약")
    print(f"{'=' * 70}")
    print(f"  통과: {len(passed_items)}/{len(items)}")
    print(f"  실패: {len(failed_items)}/{len(items)}")

    if failed_items:
        print(f"\n  실패 항목:")
        for item in failed_items:
            print(
                f"    - Component={item.get('Component')}, "
                f"dep={item.get('dep')}, alt={item.get('alt')}"
            )

    # 통과 항목 저장
    out_path = json_path.replace(".json", "_validated.json")
    with open(out_path, "w") as f:
        json.dump(passed_items, f, indent=2, ensure_ascii=False)
    print(f"\n  통과 항목 저장: {out_path}")

    return passed_items, failed_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="variable_names.json")
    args = parser.parse_args()

    print("Loading tokenizers...")
    llama_tok = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    print(f"  ✓ Llama (vocab: {llama_tok.vocab_size:,})")
    gemma_tok = AutoTokenizer.from_pretrained(GEMMA_MODEL)
    print(f"  ✓ Gemma (vocab: {gemma_tok.vocab_size:,})")

    validate(args.json, llama_tok, gemma_tok)


if __name__ == "__main__":
    main()
