#!/usr/bin/env python3
"""
Run validation, optional logit measurement, and analysis for the
reactivity ablation dataset (Ablation C: body-use; Ablation B: context-shape,
optional).

Default behavior is local-only for model/tokenizer loading. If the model stack
or local weights are unavailable, the script writes a validation report and
stops before writing logit/analysis CSVs.

Framing note: all claims and labels here stay at the behavioral level
(LD_stateform, state-form preference, declaration prior, body-use retrieval,
callback-array context). This script does not assert React-semantics or
reactivity-understanding claims.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


C_CONTEXTS = ["useEffect", "subscribe"]
B_CONTEXTS = ["useEffect", "subscribe", "plain_array", "return_array"]
CALLBACK_CONTEXTS = {"useEffect", "subscribe"}
STATEMENT_TAILS = {
    "plain_array": "  const values = [",
    "return_array": "  return [",
}

STATE_ROLES = ["dep_reactive", "alt_reactive"]
DECL_ORDERS = ["reactive_first", "stable_first"]
BODY_ORDERS = ["dep_first", "alt_first"]
BODYUSE_ORDER = ["both", "target_only", "control_only", "neither"]

MODEL_SPECS = [
    {
        "key": "llama3.2-1b",
        "label": "Llama 3.2 1B",
        "hf_id": "meta-llama/Llama-3.2-1B",
    },
    {
        "key": "gemma3-1b-pt",
        "label": "Gemma 3 1B-pt",
        "hf_id": "google/gemma-3-1b-pt",
    },
]
PRIMARY_MODEL_KEY = "llama3.2-1b"


class StructuralValidationError(RuntimeError):
    pass


class TokenValidationError(RuntimeError):
    pass


@dataclass
class Availability:
    ok: bool
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="../dataset/reactivity_ablation_dataset.json",
        help="Generated dataset JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default="reactivity_ablation",
        help="Directory for CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--skip-logits",
        action="store_true",
        help="Only run structural validation and write a validation report.",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face downloads. Default is local-only.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=45,
        help="Expected number of validated variable pairs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Bootstrap samples for Stage A CIs.",
    )
    return parser.parse_args()


def load_rows(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise StructuralValidationError("Dataset JSON must be a list of rows.")
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StructuralValidationError(message)


def body_region(prefix: str, row: dict[str, Any]) -> str:
    start = row["decl_block_end_offset"]
    condition = row["condition"]
    if condition in CALLBACK_CONTEXTS:
        end = prefix.rfind("  }, [")
    elif condition in STATEMENT_TAILS:
        end = prefix.rfind(STATEMENT_TAILS[condition])
    else:
        raise StructuralValidationError(f"unknown condition: {condition}")
    if end == -1 or end <= start:
        raise StructuralValidationError(f"{row['id']}: could not isolate body region.")
    return prefix[start:end]


def mentions(text: str, var_name: str) -> bool:
    return re.search(rf"\b{re.escape(var_name)}\b", text) is not None


REQUIRED_COLS = {
    "id",
    "pair_id",
    "ablation",
    "condition",
    "bodyuse",
    "state_role",
    "decl_order",
    "body_order",
    "dep_var",
    "alt_var",
    "target_var",
    "control_var",
    "useState_form_var",
    "const_form_var",
    "role_dep_useState_form",
    "decl_dep_first",
    "decl_block_end_offset",
    "prefix",
}


def validate_row(row: dict[str, Any]) -> None:
    missing = REQUIRED_COLS - set(row)
    require(not missing, f"{row.get('id', '<unknown>')} missing columns: {sorted(missing)}")

    prefix = row["prefix"]
    require(prefix.endswith("["), f"{row['id']} prefix does not end at '['.")
    require(not prefix.endswith("[ "), f"{row['id']} has whitespace after '['.")

    target = row["target_var"]
    control = row["control_var"]
    require(target != control, f"{row['id']} target_var and control_var match.")
    require(
        row["role_dep_useState_form"] == (row["state_role"] == "dep_reactive"),
        f"{row['id']} role_dep_useState_form disagrees with state_role.",
    )
    require(
        target == (row["dep_var"] if row["role_dep_useState_form"] else row["alt_var"]),
        f"{row['id']} target_var disagrees with role_dep_useState_form.",
    )

    decl_block = prefix[: row["decl_block_end_offset"]]
    require(
        f"useState(" in decl_block and target in decl_block,
        f"{row['id']} target (state-form) declaration missing or out of scope.",
    )
    require(
        f"const {control} = " in decl_block,
        f"{row['id']} control (const-form) declaration missing or out of scope.",
    )

    body = body_region(prefix, row)
    has_target = mentions(body, target)
    has_control = mentions(body, control)
    bodyuse = row["bodyuse"]
    if bodyuse == "both":
        require(
            mentions(body, row["dep_var"]) and mentions(body, row["alt_var"]),
            f"{row['id']} bodyuse=both must mention both dep_var and alt_var.",
        )
    elif bodyuse == "target_only":
        require(
            has_target and not has_control,
            f"{row['id']} bodyuse=target_only must mention target only.",
        )
    elif bodyuse == "control_only":
        require(
            has_control and not has_target,
            f"{row['id']} bodyuse=control_only must mention control only.",
        )
    elif bodyuse == "neither":
        require(
            not has_target and not has_control,
            f"{row['id']} bodyuse=neither must mention neither variable.",
        )
    else:
        raise StructuralValidationError(f"{row['id']} unknown bodyuse: {bodyuse}")


def validate_structure(rows: list[dict[str, Any]], expected_pairs: int) -> dict[str, Any]:
    for row in rows:
        validate_row(row)

    pair_ids = sorted({row["pair_id"] for row in rows})
    require(len(pair_ids) == expected_pairs, f"Expected {expected_pairs} pairs, got {len(pair_ids)}.")

    rows_c = [row for row in rows if row["ablation"] == "C"]
    rows_b = [row for row in rows if row["ablation"] == "B"]

    expected_c_per_pair_context = len(STATE_ROLES) * len(DECL_ORDERS) * (
        (len(BODYUSE_ORDER) - 1) + len(BODY_ORDERS)
    )
    by_pair_context_c: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows_c:
        by_pair_context_c[(row["pair_id"], row["condition"])].append(row)
    for pair_id in pair_ids:
        for context in C_CONTEXTS:
            cells = by_pair_context_c[(pair_id, context)]
            require(
                len(cells) == expected_c_per_pair_context,
                f"[{pair_id}/{context}] expected {expected_c_per_pair_context} ablation-C cells, got {len(cells)}.",
            )
            by_bodyuse = defaultdict(list)
            for row in cells:
                by_bodyuse[row["bodyuse"]].append(row)
            for bodyuse in BODYUSE_ORDER:
                group = by_bodyuse[bodyuse]
                expected = (
                    len(STATE_ROLES) * len(DECL_ORDERS) * len(BODY_ORDERS)
                    if bodyuse == "both"
                    else len(STATE_ROLES) * len(DECL_ORDERS)
                )
                require(
                    len(group) == expected,
                    f"[{pair_id}/{context}/{bodyuse}] expected {expected} cells, got {len(group)}.",
                )
                combos = {
                    (row["state_role"], row["decl_order"], row["body_order"]) for row in group
                }
                require(
                    len(combos) == expected,
                    f"[{pair_id}/{context}/{bodyuse}] role/decl(/body_order) crossing has duplicates.",
                )

    if rows_b:
        by_pair_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_b:
            by_pair_b[row["pair_id"]].append(row)
        expected_b_per_pair = len(B_CONTEXTS) * len(STATE_ROLES) * len(DECL_ORDERS)
        for pair_id in pair_ids:
            cells = by_pair_b[pair_id]
            require(
                len(cells) == expected_b_per_pair,
                f"[{pair_id}] expected {expected_b_per_pair} ablation-B cells, got {len(cells)}.",
            )
            combos = {
                (row["condition"], row["state_role"], row["decl_order"]) for row in cells
            }
            require(
                len(combos) == expected_b_per_pair,
                f"[{pair_id}] ablation-B context/role/decl crossing has duplicates.",
            )
            decl_blocks = {
                row["prefix"][: row["decl_block_end_offset"]].split("function Component", 1)[-1]
                for row in cells
                if row["state_role"] == "dep_reactive" and row["decl_order"] == "reactive_first"
            }
            require(
                len(decl_blocks) == 1,
                f"[{pair_id}] ablation-B declarations are not identical across contexts.",
            )

    return {
        "n_rows": len(rows),
        "n_pairs": len(pair_ids),
        "n_rows_c": len(rows_c),
        "n_rows_b": len(rows_b),
        "c_contexts": C_CONTEXTS,
        "b_contexts": B_CONTEXTS if rows_b else [],
        "structural_validation": "passed",
    }


def import_model_stack() -> tuple[Availability, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on local env
        return Availability(False, f"model stack unavailable: {exc}"), None, None, None
    return Availability(True), torch, AutoModelForCausalLM, AutoTokenizer


def resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode(tokenizer: Any, text: str) -> Any:
    return tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids


def token_to_str(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def target_id(
    tokenizer: Any,
    prefix: str,
    suffix: str,
    prefix_n: int,
    expected_add: int = 1,
) -> int:
    full_tokens = encode(tokenizer, prefix + suffix)
    n_tokens = int(full_tokens.shape[-1])
    if n_tokens != prefix_n + expected_add:
        raise TokenValidationError(
            f"token count mismatch: prefix_n={prefix_n}, suffix={suffix!r}, "
            f"full_n={n_tokens}, expected_add={expected_add}"
        )
    return int(full_tokens[0, -1].item())


def validate_row_tokens(tokenizer: Any, row: dict[str, Any]) -> dict[str, int]:
    prefix = row["prefix"]
    prefix_tokens = encode(tokenizer, prefix)
    prefix_n = int(prefix_tokens.shape[-1])
    last_tok = token_to_str(tokenizer, int(prefix_tokens[0, -1].item())).strip()
    if last_tok != "[":
        raise TokenValidationError(f"last prefix token is not '[': {last_tok!r}")

    dep_id = target_id(tokenizer, prefix, row["dep_var"], prefix_n)
    alt_id = target_id(tokenizer, prefix, row["alt_var"], prefix_n)
    if dep_id == alt_id:
        raise TokenValidationError("dep_id and alt_id are identical")

    target_var_id = dep_id if row["role_dep_useState_form"] else alt_id
    control_var_id = alt_id if row["role_dep_useState_form"] else dep_id

    decl_block = prefix[: row["decl_block_end_offset"]]
    decl_n = int(encode(tokenizer, decl_block).shape[-1])

    return {
        "n_tokens": prefix_n,
        "decl_n_tokens": decl_n,
        "decl_to_bracket_token_distance": prefix_n - decl_n,
        "dep_id": dep_id,
        "alt_id": alt_id,
        "target_id": target_var_id,
        "control_id": control_var_id,
    }


def validate_tokens_for_models(
    rows: list[dict[str, Any]],
    tokenizers: dict[str, Any],
) -> tuple[dict[str, dict[str, dict[str, int]]], list[dict[str, str]], set[str]]:
    token_info: dict[str, dict[str, dict[str, int]]] = {}
    failures: list[dict[str, str]] = []
    failed_pairs: set[str] = set()

    for model_key, tokenizer in tokenizers.items():
        token_info[model_key] = {}
        for row in rows:
            try:
                token_info[model_key][row["id"]] = validate_row_tokens(tokenizer, row)
            except TokenValidationError as exc:
                failures.append(
                    {
                        "model": model_key,
                        "pair_id": row["pair_id"],
                        "row_id": row["id"],
                        "reason": str(exc),
                    }
                )
                failed_pairs.add(row["pair_id"])

    return token_info, failures, failed_pairs


def load_tokenizers(auto_tokenizer: Any, local_files_only: bool) -> tuple[Availability, dict[str, Any]]:
    tokenizers: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        try:
            tokenizers[spec["key"]] = auto_tokenizer.from_pretrained(
                spec["hf_id"],
                local_files_only=local_files_only,
            )
        except Exception as exc:  # pragma: no cover - depends on local env
            return (
                Availability(
                    False,
                    f"tokenizer unavailable for {spec['key']} ({spec['hf_id']}): {exc}",
                ),
                {},
            )
    return Availability(True), tokenizers


def torch_dtype_for_device(torch: Any, device: str) -> Any:
    # float16 has a narrow dynamic range and is known to overflow to NaN/Inf
    # mid-forward-pass on models with large activation norms (Gemma in
    # particular). bfloat16 has float32's exponent range and is the safe
    # reduced-precision choice; fall back to float32 where bf16 isn't
    # available rather than risking silent NaN logits.
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model(torch: Any, auto_model: Any, spec: dict[str, str], device: str, local_files_only: bool) -> Any:
    model = auto_model.from_pretrained(
        spec["hf_id"],
        local_files_only=local_files_only,
        torch_dtype=torch_dtype_for_device(torch, device),
    )
    model.eval()
    model.to(device)
    return model


def ld_stateform_from_d(role_dep_use_state_form: bool, d_value: float) -> float:
    return (1.0 if role_dep_use_state_form else -1.0) * d_value


def measure_logits(
    rows: list[dict[str, Any]],
    tokenizers: dict[str, Any],
    token_info: dict[str, dict[str, dict[str, int]]],
    torch: Any,
    auto_model: Any,
    device: str,
    local_files_only: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        model_key = spec["key"]
        tokenizer = tokenizers[model_key]
        model = load_model(torch, auto_model, spec, device, local_files_only)

        for row in rows:
            info = token_info[model_key][row["id"]]
            inputs = tokenizer(row["prefix"], return_tensors="pt", add_special_tokens=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
            final_logits = output.logits[0, -1].detach().float().cpu()

            dep_id = info["dep_id"]
            alt_id = info["alt_id"]
            target_var_id = info["target_id"]
            control_var_id = info["control_id"]

            logit_dep = float(final_logits[dep_id].item())
            logit_alt = float(final_logits[alt_id].item())
            logit_target = float(final_logits[target_var_id].item())
            logit_control = float(final_logits[control_var_id].item())
            d_value = logit_dep - logit_alt
            ld_stateform = ld_stateform_from_d(bool(row["role_dep_useState_form"]), d_value)
            direct_ld = logit_target - logit_control

            records.append(
                {
                    "model": model_key,
                    "ablation": row["ablation"],
                    "condition": row["condition"],
                    "bodyuse": row["bodyuse"],
                    "pair_id": row["pair_id"],
                    "cell_id": row["id"],
                    "state_role": row["state_role"],
                    "decl_order": row["decl_order"],
                    "body_order": row["body_order"],
                    "dep_var": row["dep_var"],
                    "alt_var": row["alt_var"],
                    "target_var": row["target_var"],
                    "control_var": row["control_var"],
                    "role_dep_useState_form": bool(row["role_dep_useState_form"]),
                    "decl_dep_first": bool(row["decl_dep_first"]),
                    "pos_dep_first": row["pos_dep_first"],
                    "n_tokens": info["n_tokens"],
                    "decl_to_bracket_token_distance": info["decl_to_bracket_token_distance"],
                    "dep_id": dep_id,
                    "alt_id": alt_id,
                    "target_id": target_var_id,
                    "control_id": control_var_id,
                    "logit_dep": logit_dep,
                    "logit_alt": logit_alt,
                    "logit_target": logit_target,
                    "logit_control": logit_control,
                    "D": d_value,
                    "LD_stateform": ld_stateform,
                    "LD_reactive": ld_stateform,
                    "LD_stateform_direct": direct_ld,
                }
            )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    assert_finite_logits(df)
    assert_resign_sanity(df)
    return df


def assert_finite_logits(df: pd.DataFrame) -> None:
    """Fail loudly on non-finite logits instead of letting NaN propagate.

    NaN compared with `> 0` is False, so a silently-NaN LD_stateform column
    shows up downstream as frac_pos == 0.0 with every summary stat NaN --
    indistinguishable at a glance from "no effect" unless caught here.
    """
    cols = ["logit_dep", "logit_alt", "logit_target", "logit_control", "D", "LD_stateform"]
    bad = df[~np.isfinite(df[cols]).all(axis=1)]
    if not bad.empty:
        by_model = bad.groupby("model").size().to_dict()
        sample = bad[["model", "cell_id"]].head(5).to_dict("records")
        raise RuntimeError(
            f"Non-finite logits (NaN/Inf) in {len(bad)}/{len(df)} rows, by model: {by_model}. "
            f"Sample: {sample}. Likely cause: forward-pass overflow in the model's compute "
            f"dtype (e.g. float16) -- see torch_dtype_for_device."
        )


def assert_resign_sanity(df: pd.DataFrame) -> None:
    check = (df["LD_stateform"] - df["LD_stateform_direct"]).abs()
    max_abs = float(check.max())
    if max_abs > 1e-5:
        bad = df.loc[check.idxmax(), ["model", "condition", "cell_id"]].to_dict()
        raise AssertionError(f"LD_stateform re-sign sanity check failed: max_abs={max_abs}, row={bad}")


def one_sided_t_pvalue(t_stat: float, p_two: float) -> float:
    if math.isnan(t_stat) or math.isnan(p_two):
        return math.nan
    return p_two / 2.0 if t_stat >= 0 else 1.0 - (p_two / 2.0)


def bootstrap_ci(values: np.ndarray, statistic: str, n_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    draws = rng.choice(values, size=(n_samples, len(values)), replace=True)
    if statistic == "mean":
        estimates = draws.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(draws, axis=1)
    else:
        raise ValueError(statistic)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def cohens_d(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return math.nan
    return float(values.mean() / sd)


def one_sample_rows(
    pair_means: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in pair_means.groupby(group_cols):
        keys = key if isinstance(key, tuple) else (key,)
        values = group[value_col].to_numpy(dtype=float)
        t_stat, p_two = stats.ttest_1samp(values, 0.0)
        try:
            wilcoxon = stats.wilcoxon(values, alternative="greater", zero_method="wilcox")
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_p = math.nan
        mean_lo, mean_hi = bootstrap_ci(values, "mean", bootstrap_samples, rng)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_pairs": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "frac_pos": float((values > 0).mean()),
                "t_stat": float(t_stat),
                "t_p_onesided_greater": one_sided_t_pvalue(float(t_stat), float(p_two)),
                "t_p_twosided": float(p_two),
                "wilcoxon_p_onesided_greater": wilcoxon_p,
                "cohend": cohens_d(values),
                "mean_boot_ci95_low": mean_lo,
                "mean_boot_ci95_high": mean_hi,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_contrast(
    pair_means: pd.DataFrame,
    group_cols: list[str],
    baseline_filter: dict[str, Any],
    value_col: str,
) -> pd.DataFrame:
    """Paired (by pair_id) contrast of each group against a baseline slice."""
    other_cols = [c for c in group_cols if c not in baseline_filter]
    mask = np.ones(len(pair_means), dtype=bool)
    for k, v in baseline_filter.items():
        mask &= pair_means[k] == v
    base = pair_means[mask][["pair_id", value_col] + other_cols].rename(
        columns={value_col: "baseline_value"}
    )
    base = base.drop(columns=[c for c in other_cols if c in base.columns and c != "pair_id"])

    rows: list[dict[str, Any]] = []
    for key, group in pair_means.groupby(group_cols):
        keys = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_cols, keys))
        merged = group[["pair_id", value_col]].merge(base, on="pair_id", how="inner")
        diff = (merged[value_col] - merged["baseline_value"]).to_numpy(dtype=float)
        is_baseline = all(key_dict.get(k) == v for k, v in baseline_filter.items())
        row = dict(key_dict)
        if is_baseline or len(diff) == 0:
            row.update(
                {
                    "n": len(diff),
                    "mean_diff_vs_baseline": 0.0,
                    "t": math.nan,
                    "p_twosided": math.nan,
                    "cohend": math.nan,
                }
            )
        else:
            t_stat, p_two = stats.ttest_1samp(diff, 0.0)
            row.update(
                {
                    "n": len(diff),
                    "mean_diff_vs_baseline": float(diff.mean()),
                    "t": float(t_stat),
                    "p_twosided": float(p_two),
                    "cohend": cohens_d(diff),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def stage_a_ablation_c(df_c: pd.DataFrame, bootstrap_samples: int) -> dict[str, pd.DataFrame]:
    pair_means = (
        df_c.groupby(["model", "condition", "bodyuse", "pair_id"], as_index=False)
        .agg(mean_LD_stateform=("LD_stateform", "mean"))
    )
    rng = np.random.default_rng(12345)
    one_sample = one_sample_rows(
        pair_means, ["model", "condition", "bodyuse"], "mean_LD_stateform", bootstrap_samples, rng
    )

    paired_frames = []
    for model, model_means in pair_means.groupby("model"):
        for context in C_CONTEXTS:
            ctx_means = model_means[model_means["condition"] == context]
            contrast = paired_contrast(
                ctx_means,
                ["model", "condition", "bodyuse"],
                {"model": model, "condition": context, "bodyuse": "both"},
                "mean_LD_stateform",
            )
            paired_frames.append(contrast)
    paired_vs_both = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()

    headline = one_sample[one_sample["bodyuse"] == "control_only"].copy()
    return {
        "pair_means": pair_means,
        "one_sample": one_sample,
        "paired_vs_both": paired_vs_both,
        "headline_control_only": headline,
    }


def stage_a_ablation_b(df_b: pd.DataFrame, bootstrap_samples: int) -> dict[str, pd.DataFrame]:
    pair_means = (
        df_b.groupby(["model", "condition", "pair_id"], as_index=False)
        .agg(mean_LD_stateform=("LD_stateform", "mean"))
    )
    rng = np.random.default_rng(12345)
    one_sample = one_sample_rows(pair_means, ["model", "condition"], "mean_LD_stateform", bootstrap_samples, rng)

    paired_frames = []
    for model, model_means in pair_means.groupby("model"):
        contrast = paired_contrast(
            model_means, ["model", "condition"], {"model": model, "condition": "useEffect"}, "mean_LD_stateform"
        )
        paired_frames.append(contrast)
    paired_vs_useeffect = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()

    return {"pair_means": pair_means, "one_sample": one_sample, "paired_vs_useEffect": paired_vs_useeffect}


def center_distance(values: pd.Series) -> pd.Series:
    """Mean-center the decl-to-bracket distance covariate.

    Deliberately not z-scored: scipy.stats.zscore hits a catastrophic-
    cancellation failure mode (returns NaN for every row) when the values are
    tightly clustered, which this covariate often is within a single model's
    subset. Mean-centering only needs a subtraction, so it can't divide by a
    near-zero std and can't produce NaN/inf.
    """
    x = values.astype(float)
    return x - x.mean()


def regression_c(df_c: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm

    both = df_c[df_c["bodyuse"] == "both"].copy()
    rows: list[dict[str, Any]] = []
    for model, group in both.groupby("model"):
        work = group.copy()
        work["pos_dep_first_code"] = np.where(work["pos_dep_first"], 1.0, -1.0)
        work["role_dep_useState_form_code"] = np.where(work["role_dep_useState_form"], 1.0, -1.0)
        work["decl_dep_first_code"] = np.where(work["decl_dep_first"], 1.0, -1.0)
        work["condition_subscribe"] = (work["condition"] == "subscribe").astype(float)
        work["role_x_subscribe"] = work["role_dep_useState_form_code"] * work["condition_subscribe"]
        work["distance_c"] = center_distance(work["decl_to_bracket_token_distance"])

        feature_cols = [
            "pos_dep_first_code",
            "role_dep_useState_form_code",
            "decl_dep_first_code",
            "condition_subscribe",
            "role_x_subscribe",
            "distance_c",
        ]
        x = sm.add_constant(work[feature_cols], has_constant="add")
        y = work["D"].astype(float)
        result = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": work["pair_id"]})
        conf = result.conf_int()
        for term in result.params.index:
            rows.append(
                {
                    "model": model,
                    "dependent_var": "D",
                    "subset": "bodyuse=both",
                    "term": term,
                    "estimate": float(result.params[term]),
                    "std_err": float(result.bse[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "p_value": float(result.pvalues[term]),
                }
            )
    return pd.DataFrame(rows)


def regression_b(df_b: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm

    rows: list[dict[str, Any]] = []
    for model, group in df_b.groupby("model"):
        work = group.copy()
        work["role_dep_useState_form_code"] = np.where(work["role_dep_useState_form"], 1.0, -1.0)
        work["decl_dep_first_code"] = np.where(work["decl_dep_first"], 1.0, -1.0)
        work["distance_c"] = center_distance(work["decl_to_bracket_token_distance"])

        feature_cols = ["role_dep_useState_form_code", "decl_dep_first_code", "distance_c"]
        for context in B_CONTEXTS[1:]:
            col = f"condition_{context}"
            work[col] = (work["condition"] == context).astype(float)
            feature_cols.append(col)
        for context in B_CONTEXTS[1:]:
            col = f"role_x_{context}"
            work[col] = work["role_dep_useState_form_code"] * (work["condition"] == context).astype(float)
            feature_cols.append(col)

        x = sm.add_constant(work[feature_cols], has_constant="add")
        y = work["D"].astype(float)
        result = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": work["pair_id"]})
        conf = result.conf_int()
        for term in result.params.index:
            rows.append(
                {
                    "model": model,
                    "dependent_var": "D",
                    "term": term,
                    "estimate": float(result.params[term]),
                    "std_err": float(result.bse[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "p_value": float(result.pvalues[term]),
                }
            )
    return pd.DataFrame(rows)


def make_wide(df: pd.DataFrame, extra_index: list[str]) -> pd.DataFrame:
    work = df.copy()
    work["combo"] = (
        "sr=" + work["state_role"] + "|do=" + work["decl_order"] + "|bo=" + work["body_order"].fillna("na")
    )
    pivot = work.pivot_table(
        index=["model", "condition", *extra_index, "pair_id"],
        columns="combo",
        values=["D", "LD_stateform", "logit_target", "logit_control"],
        aggfunc="first",
    )
    pivot.columns = [f"{value}__{combo}" for value, combo in pivot.columns]
    return pivot.reset_index()


PRE_REGISTERED_PREDICTIONS = """\
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
"""

B_SCOPE_NOTE = """\
## Scope note (Ablation B, optional)

Headline framing is not fixed in advance. If the effect holds in `useEffect` and
`subscribe` but drops in `plain_array`/`return_array`, that reads as gated by
callback-array context. If it holds across all four, that reads as a broader
array-completion prior not specific to callback wrapping. The phrase "beyond dependency
arrays" is licensed only by the second pattern. Until resolved, all claims here are scoped
to callback-array completion, not dependency-array completion specifically.
"""


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> list[str]:
    if df.empty:
        return ["(empty)"]
    sub = df[columns].head(max_rows).copy()
    return sub.to_markdown(index=False).splitlines()


def write_validation_report(out_dir: Path, structural: dict[str, Any], availability: Availability,
                             token_failures: list[dict[str, str]] | None = None,
                             failed_pairs: set[str] | None = None,
                             token_validation_status: str = "not run") -> None:
    token_failures = token_failures or []
    failed_pairs = failed_pairs or set()
    lines = [
        "# Reactivity Ablation Validation Report",
        "",
        "## Dataset",
        "",
        f"- rows: {structural['n_rows']} (C: {structural['n_rows_c']}, B: {structural['n_rows_b']})",
        f"- pairs: {structural['n_pairs']}",
        f"- ablation C contexts: {', '.join(structural['c_contexts'])}",
        f"- ablation B contexts: {', '.join(structural['b_contexts']) if structural['b_contexts'] else '(not run)'}",
        f"- structural validation: {structural['structural_validation']}",
        "",
        "## Token/Model Availability",
        "",
    ]
    if availability.ok:
        lines.append("- local model stack/tokenizers: available")
    else:
        lines.append("- local model stack/tokenizers: unavailable")
        lines.append(f"- reason: `{availability.reason}`")
    lines.extend(
        [
            f"- token validation: {token_validation_status}",
            f"- token-validation failures: {len(token_failures)}",
            f"- failed pairs flagged for dropping: {len(failed_pairs)}",
        ]
    )
    if token_failures[:10]:
        lines.extend(["", "First token-validation failures:"])
        for failure in token_failures[:10]:
            lines.append(f"- {failure['model']} {failure['row_id']}: {failure['reason']}")
    lines.extend(
        [
            "",
            "## TODO",
            "",
            "Run the logit step in an environment with local Llama 3.2 1B and Gemma 3 1B-pt weights",
            "(run from inside this directory so the relative dataset path resolves):",
            "",
            "```bash",
            "cd reactivity_ablation && python3 run_reactivity_ablation.py",
            "```",
        ]
    )
    (out_dir / "reactivity_ablation_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_stub(out_dir: Path, structural: dict[str, Any], availability: Availability,
                        token_validation_status: str) -> None:
    lines = [
        "# Reactivity Ablation Summary",
        "",
        "## Dataset",
        "",
        f"- rows: {structural['n_rows']} (C: {structural['n_rows_c']}, B: {structural['n_rows_b']})",
        f"- pairs: {structural['n_pairs']}",
        f"- structural validation: {structural['structural_validation']}",
        f"- token validation: {token_validation_status}",
        "",
        "## Sign Convention",
        "",
        "`D = logit(dep) - logit(alt)`.",
        "",
        "`LD_stateform = (role_dep_useState_form ? +1 : -1) * D`.",
        "",
        PRE_REGISTERED_PREDICTIONS,
        B_SCOPE_NOTE,
        "## Status",
        "",
        "Stopped before logit measurement.",
        "",
        f"Reason: `{availability.reason}`",
        "",
        "TODO: rerun this script in the GPU/model environment with local weights.",
    ]
    (out_dir / "reactivity_ablationC_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_c_outputs(out_dir: Path, df_c: pd.DataFrame, bootstrap_samples: int) -> dict[str, pd.DataFrame]:
    stage_a = stage_a_ablation_c(df_c, bootstrap_samples)
    reg = regression_c(df_c)
    wide = make_wide(df_c, ["bodyuse"])

    df_c.to_csv(out_dir / "reactivity_ablationC_logits_3tokens.csv", index=False)
    wide.to_csv(out_dir / "reactivity_ablationC_wide_per_sample.csv", index=False)
    stage_a["one_sample"].to_csv(out_dir / "reactivity_ablationC_pairlevel_test.csv", index=False)
    reg.to_csv(out_dir / "reactivity_ablationC_regression.csv", index=False)

    lines = [
        "# Reactivity Ablation C Summary (body-use)",
        "",
        "## Dataset",
        "",
        f"- rows: {len(df_c)}",
        f"- contexts: {', '.join(C_CONTEXTS)}",
        "",
        "## Sign Convention",
        "",
        "`D = logit(dep) - logit(alt)`. `LD_stateform = (role_dep_useState_form ? +1 : -1) * D`.",
        "",
        PRE_REGISTERED_PREDICTIONS,
        "## Headline: control_only one-sample test vs 0 (decision cell)",
        "",
    ]
    lines.extend(
        markdown_table(
            stage_a["headline_control_only"],
            ["model", "condition", "n_pairs", "mean", "median", "frac_pos",
             "mean_boot_ci95_low", "mean_boot_ci95_high", "t_p_onesided_greater", "cohend"],
        )
    )
    lines.extend(["", "## Stage A: all bodyuse cells, pair-level mean", ""])
    lines.extend(
        markdown_table(
            stage_a["one_sample"],
            ["model", "condition", "bodyuse", "n_pairs", "mean", "median", "frac_pos",
             "mean_boot_ci95_low", "mean_boot_ci95_high", "cohend"],
            max_rows=80,
        )
    )
    lines.extend(["", "## Ordering check: bodyuse vs. `both` (paired by pair_id)", ""])
    lines.extend(
        markdown_table(
            stage_a["paired_vs_both"],
            ["model", "condition", "bodyuse", "n", "mean_diff_vs_baseline", "t", "p_twosided", "cohend"],
            max_rows=80,
        )
    )
    lines.extend(["", "## Optional regression (bodyuse=both subset, decl_to_bracket distance as covariate)", ""])
    key_terms = reg[reg["term"].str.contains("role_dep_useState_form|distance_c", regex=True)]
    lines.extend(
        markdown_table(
            key_terms,
            ["model", "term", "estimate", "std_err", "ci_low", "ci_high", "p_value"],
            max_rows=40,
        )
    )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `reactivity_ablationC_logits_3tokens.csv`",
            "- `reactivity_ablationC_wide_per_sample.csv`",
            "- `reactivity_ablationC_pairlevel_test.csv`",
            "- `reactivity_ablationC_regression.csv`",
        ]
    )
    (out_dir / "reactivity_ablationC_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stage_a


def write_ablation_b_outputs(out_dir: Path, df_b: pd.DataFrame, bootstrap_samples: int) -> dict[str, pd.DataFrame]:
    stage_a = stage_a_ablation_b(df_b, bootstrap_samples)
    reg = regression_b(df_b)
    wide = make_wide(df_b, [])

    df_b.to_csv(out_dir / "reactivity_ablationB_logits_3tokens.csv", index=False)
    wide.to_csv(out_dir / "reactivity_ablationB_wide_per_sample.csv", index=False)
    stage_a["one_sample"].to_csv(out_dir / "reactivity_ablationB_pairlevel_test.csv", index=False)
    reg.to_csv(out_dir / "reactivity_ablationB_regression.csv", index=False)

    lines = [
        "# Reactivity Ablation B Summary (context-shape, optional)",
        "",
        "## Dataset",
        "",
        f"- rows: {len(df_b)}",
        f"- contexts: {', '.join(B_CONTEXTS)}",
        "",
        B_SCOPE_NOTE,
        "## Stage A: pair-level mean per context",
        "",
    ]
    lines.extend(
        markdown_table(
            stage_a["one_sample"],
            ["model", "condition", "n_pairs", "mean", "median", "frac_pos",
             "mean_boot_ci95_low", "mean_boot_ci95_high", "cohend"],
        )
    )
    lines.extend(["", "## Paired contrast vs. useEffect baseline", ""])
    lines.extend(
        markdown_table(
            stage_a["paired_vs_useEffect"],
            ["model", "condition", "n", "mean_diff_vs_baseline", "t", "p_twosided", "cohend"],
        )
    )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `reactivity_ablationB_logits_3tokens.csv`",
            "- `reactivity_ablationB_wide_per_sample.csv`",
            "- `reactivity_ablationB_pairlevel_test.csv`",
            "- `reactivity_ablationB_regression.csv`",
        ]
    )
    (out_dir / "reactivity_ablationB_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stage_a


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_path)
    structural = validate_structure(rows, args.expected_pairs)

    def stop_before_logits(availability: Availability, token_status: str,
                            token_failures: list[dict[str, str]] | None = None,
                            failed_pairs: set[str] | None = None) -> int:
        write_validation_report(out_dir, structural, availability, token_failures, failed_pairs, token_status)
        write_summary_stub(out_dir, structural, availability, token_status)
        print(f"Structural validation passed; stopped before logits: {availability.reason}")
        return 0

    if args.skip_logits:
        return stop_before_logits(Availability(False, "--skip-logits was set"), "not run (--skip-logits)")

    availability, torch, auto_model, auto_tokenizer = import_model_stack()
    if not availability.ok:
        return stop_before_logits(availability, "not run (model stack unavailable)")

    local_files_only = not args.allow_downloads
    tok_availability, tokenizers = load_tokenizers(auto_tokenizer, local_files_only)
    if not tok_availability.ok:
        return stop_before_logits(tok_availability, "not run (tokenizers unavailable)")

    token_info, token_failures, failed_pairs = validate_tokens_for_models(rows, tokenizers)
    valid_rows = [row for row in rows if row["pair_id"] not in failed_pairs]
    if not valid_rows:
        availability = Availability(False, "all pairs failed tokenizer validation")
        return stop_before_logits(availability, "failed", token_failures, failed_pairs)

    device = resolve_device(torch, args.device)
    try:
        df = measure_logits(valid_rows, tokenizers, token_info, torch, auto_model, device, local_files_only)
    except Exception as exc:  # pragma: no cover - depends on local env
        availability = Availability(False, f"model weights unavailable or load failed: {exc}")
        status = "passed" if not token_failures else "passed after dropping flagged pairs"
        return stop_before_logits(availability, status, token_failures, failed_pairs)

    df_c = df[df["ablation"] == "C"].copy()
    df_b = df[df["ablation"] == "B"].copy()

    write_ablation_c_outputs(out_dir, df_c, args.bootstrap_samples)
    if not df_b.empty:
        write_ablation_b_outputs(out_dir, df_b, args.bootstrap_samples)

    status = "passed" if not token_failures else "passed after dropping flagged pairs"
    write_validation_report(out_dir, structural, Availability(True), token_failures, failed_pairs, status)
    print(f"Wrote ablation outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
