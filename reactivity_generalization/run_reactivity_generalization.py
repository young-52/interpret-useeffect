#!/usr/bin/env python3
"""
Run validation, optional logit measurement, and analysis for the
reactivity generalization dataset.

Default behavior is local-only for model/tokenizer loading. If the model stack
or local weights are unavailable, the script writes a validation report and
stops before writing logit/analysis CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


CONDITIONS = [
    "useEffect",
    "useLayoutEffect",
    "alias",
    "alias_ctrl",
    "subscribe",
]

STATE_ROLES = ["dep_reactive", "alt_reactive"]
DECL_ORDERS = ["reactive_first", "stable_first"]
BODY_ORDERS = ["dep_first", "alt_first"]
EXPECTED_COMBOS = {
    (state_role, decl_order, body_order)
    for state_role in STATE_ROLES
    for decl_order in DECL_ORDERS
    for body_order in BODY_ORDERS
}

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
        default="../dataset/reactivity_generalization_dataset.json",
        help="Generated dataset JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default="reactivity_generalization",
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


def callback_body(prefix: str) -> str:
    start = prefix.rfind("=> {")
    end = prefix.rfind("  }, [")
    if start == -1 or end == -1 or end <= start:
        raise StructuralValidationError("Could not isolate callback body.")
    return prefix[start:end]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StructuralValidationError(message)


def validate_structure(
    rows: list[dict[str, Any]], expected_pairs: int
) -> dict[str, Any]:
    required_cols = {
        "id",
        "pair_id",
        "condition",
        "state_role",
        "decl_order",
        "body_order",
        "dep_var",
        "alt_var",
        "useState_form_var",
        "const_form_var",
        "role_dep_useState_form",
        "pos_dep_first",
        "decl_dep_first",
        "reactive_body_pos",
        "prefix",
    }

    for row in rows:
        missing = required_cols - set(row)
        require(not missing, f"{row.get('id', '<unknown>')} missing columns: {sorted(missing)}")
        prefix = row["prefix"]
        require(prefix.endswith("["), f"{row['id']} prefix does not end at '['.")
        require(not prefix.endswith("[ "), f"{row['id']} has whitespace after '['.")

        body = callback_body(prefix)
        dep = row["dep_var"]
        alt = row["alt_var"]
        require(f"${{{dep}}}" in body, f"{row['id']} dep var is absent from body.")
        require(f"${{{alt}}}" in body, f"{row['id']} alt var is absent from body.")
        require(
            row["useState_form_var"] != row["const_form_var"],
            f"{row['id']} has identical useState/const form vars.",
        )
        require(
            row["role_dep_useState_form"] == (row["state_role"] == "dep_reactive"),
            f"{row['id']} role_dep_useState_form disagrees with state_role.",
        )
        require(
            row["pos_dep_first"] == (row["body_order"] == "dep_first"),
            f"{row['id']} pos_dep_first disagrees with body_order.",
        )

    pair_ids = sorted({row["pair_id"] for row in rows})
    require(
        len(pair_ids) == expected_pairs,
        f"Expected {expected_pairs} pairs, got {len(pair_ids)}.",
    )
    expected_total = expected_pairs * len(CONDITIONS) * len(EXPECTED_COMBOS)
    require(
        len(rows) == expected_total,
        f"Expected {expected_total} rows, got {len(rows)}.",
    )

    by_pair_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair_condition[(row["pair_id"], row["condition"])].append(row)
        by_pair[row["pair_id"]].append(row)

    for pair_id in pair_ids:
        seen_conditions = {row["condition"] for row in by_pair[pair_id]}
        require(
            seen_conditions == set(CONDITIONS),
            f"{pair_id} condition set mismatch: {sorted(seen_conditions)}",
        )
        pos_counts = Counter(row["reactive_body_pos"] for row in by_pair[pair_id])
        require(
            pos_counts["first"] == 20 and pos_counts["second"] == 20,
            f"{pair_id} overall reactive_body_pos is not 20/20: {dict(pos_counts)}",
        )

    for pair_id in pair_ids:
        for condition in CONDITIONS:
            cells = by_pair_condition[(pair_id, condition)]
            require(
                len(cells) == 8,
                f"{pair_id}/{condition} expected 8 cells, got {len(cells)}.",
            )
            combos = {
                (row["state_role"], row["decl_order"], row["body_order"])
                for row in cells
            }
            require(
                combos == EXPECTED_COMBOS,
                f"{pair_id}/{condition} 2x2x2 mismatch.",
            )
            pos_counts = Counter(row["reactive_body_pos"] for row in cells)
            require(
                pos_counts["first"] == 4 and pos_counts["second"] == 4,
                f"{pair_id}/{condition} reactive_body_pos is not 4/4: {dict(pos_counts)}",
            )
            decl_body = {
                (row["decl_order"], row["body_order"])
                for row in cells
            }
            require(
                decl_body
                == {
                    (decl_order, body_order)
                    for decl_order in DECL_ORDERS
                    for body_order in BODY_ORDERS
                },
                f"{pair_id}/{condition} decl/body crossing mismatch.",
            )

    useeffect_rows = [row for row in rows if row["condition"] == "useEffect"]
    useeffect_pilot_ids = sorted({row.get("pilot_cell_id") for row in useeffect_rows})
    require(
        len(useeffect_rows) == expected_pairs * 8,
        f"useEffect replication rows mismatch: {len(useeffect_rows)}.",
    )
    require(
        len(useeffect_pilot_ids) == expected_pairs * 8,
        "useEffect rows do not preserve unique pilot_cell_id values.",
    )

    return {
        "n_rows": len(rows),
        "n_pairs": len(pair_ids),
        "conditions": CONDITIONS,
        "rows_per_condition": dict(Counter(row["condition"] for row in rows)),
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

    use_state_id = dep_id if row["role_dep_useState_form"] else alt_id
    const_id = alt_id if row["role_dep_useState_form"] else dep_id
    return {
        "n_tokens": prefix_n,
        "dep_id": dep_id,
        "alt_id": alt_id,
        "useState_form_id": use_state_id,
        "const_form_id": const_id,
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


def load_tokenizers(
    auto_tokenizer: Any,
    local_files_only: bool,
) -> tuple[Availability, dict[str, Any]]:
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
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model(
    torch: Any,
    auto_model: Any,
    spec: dict[str, str],
    device: str,
    local_files_only: bool,
) -> Any:
    model = auto_model.from_pretrained(
        spec["hf_id"],
        local_files_only=local_files_only,
        torch_dtype=torch_dtype_for_device(torch, device),
    )
    model.eval()
    model.to(device)
    return model


def close_token_info(tokenizer: Any, row: dict[str, Any], prefix_n: int) -> tuple[int | None, str]:
    try:
        close_id = target_id(
            tokenizer,
            row["prefix"],
            row["dep_var"] + row.get("close_target", "]"),
            prefix_n,
            expected_add=2,
        )
        return close_id, ""
    except TokenValidationError as exc:
        return None, str(exc)


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
            inputs = tokenizer(
                row["prefix"],
                return_tensors="pt",
                add_special_tokens=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
            final_logits = output.logits[0, -1].detach().float().cpu()

            dep_id = info["dep_id"]
            alt_id = info["alt_id"]
            use_state_id = info["useState_form_id"]
            const_id = info["const_form_id"]

            close_id, close_error = close_token_info(tokenizer, row, info["n_tokens"])
            logit_close = (
                float(final_logits[close_id].item()) if close_id is not None else math.nan
            )

            logit_dep = float(final_logits[dep_id].item())
            logit_alt = float(final_logits[alt_id].item())
            logit_use_state = float(final_logits[use_state_id].item())
            logit_const = float(final_logits[const_id].item())
            d_value = logit_dep - logit_alt
            ld_stateform = ld_stateform_from_d(
                bool(row["role_dep_useState_form"]),
                d_value,
            )
            direct_ld = logit_use_state - logit_const

            records.append(
                {
                    "model": model_key,
                    "condition": row["condition"],
                    "pair_id": row["pair_id"],
                    "cell_id": row["id"],
                    "pilot_cell_id": row.get("pilot_cell_id", ""),
                    "state_role": row["state_role"],
                    "decl_order": row["decl_order"],
                    "body_order": row["body_order"],
                    "dep_var": row["dep_var"],
                    "alt_var": row["alt_var"],
                    "useState_form_var": row["useState_form_var"],
                    "const_form_var": row["const_form_var"],
                    "reactive_var": row["reactive_var"],
                    "stable_var": row["stable_var"],
                    "role_dep_useState_form": bool(row["role_dep_useState_form"]),
                    "pos_dep_first": bool(row["pos_dep_first"]),
                    "decl_dep_first": bool(row["decl_dep_first"]),
                    "reactive_body_pos": row["reactive_body_pos"],
                    "useState_form_body_pos": row["useState_form_body_pos"],
                    "prefix": row["prefix"],
                    "n_tokens": info["n_tokens"],
                    "dep_id": dep_id,
                    "alt_id": alt_id,
                    "useState_form_id": use_state_id,
                    "const_form_id": const_id,
                    "close_id": close_id if close_id is not None else "",
                    "logit_dep": logit_dep,
                    "logit_alt": logit_alt,
                    "logit_useState_form": logit_use_state,
                    "logit_const_form": logit_const,
                    "logit_reactive": logit_use_state,
                    "logit_stable": logit_const,
                    "logit_close": logit_close,
                    "D": d_value,
                    "LD_stateform": ld_stateform,
                    "LD_reactive": ld_stateform,
                    "LD_stateform_direct": direct_ld,
                    "LD_stateform_vs_close": logit_use_state - logit_close
                    if not math.isnan(logit_close)
                    else math.nan,
                    "LD_constform_vs_close": logit_const - logit_close
                    if not math.isnan(logit_close)
                    else math.nan,
                    "close_error": close_error,
                }
            )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    assert_resign_sanity(df)
    return df


def assert_resign_sanity(df: pd.DataFrame) -> None:
    check = (df["LD_stateform"] - df["LD_stateform_direct"]).abs()
    max_abs = float(check.max())
    if max_abs > 1e-5:
        bad = df.loc[check.idxmax(), ["model", "condition", "cell_id"]].to_dict()
        raise AssertionError(
            f"LD_stateform re-sign sanity check failed: max_abs={max_abs}, row={bad}"
        )


def one_sided_t_pvalue(t_stat: float, p_two: float, alternative: str = "greater") -> float:
    if alternative != "greater":
        raise ValueError("Only greater alternative is used here.")
    if math.isnan(t_stat) or math.isnan(p_two):
        return math.nan
    return p_two / 2.0 if t_stat >= 0 else 1.0 - (p_two / 2.0)


def bootstrap_ci(
    values: np.ndarray,
    statistic: str,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
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


def stage_a_pairlevel(df: pd.DataFrame, bootstrap_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_means = (
        df.groupby(["model", "condition", "pair_id"], as_index=False)
        .agg(mean_LD_stateform_pair=("LD_stateform", "mean"), n_cells=("LD_stateform", "size"))
    )

    rng = np.random.default_rng(12345)
    rows: list[dict[str, Any]] = []
    for (model, condition), group in pair_means.groupby(["model", "condition"]):
        values = group["mean_LD_stateform_pair"].to_numpy(dtype=float)
        t_stat, p_two = stats.ttest_1samp(values, 0.0)
        try:
            wilcoxon = stats.wilcoxon(values, alternative="greater", zero_method="wilcox")
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = math.nan
            wilcoxon_p = math.nan
        mean_ci_low, mean_ci_high = bootstrap_ci(
            values, "mean", bootstrap_samples, rng
        )
        median_ci_low, median_ci_high = bootstrap_ci(
            values, "median", bootstrap_samples, rng
        )

        row = {
            "model": model,
            "condition": condition,
            "n_pairs": len(values),
            "pair_mean_mean": float(values.mean()),
            "pair_mean_median": float(np.median(values)),
            "pair_mean_sd": float(values.std(ddof=1)),
            "frac_pair_mean_pos": float((values > 0).mean()),
            "t_stat": float(t_stat),
            "t_p_onesided_greater": one_sided_t_pvalue(float(t_stat), float(p_two)),
            "t_p_twosided": float(p_two),
            "cohend": cohens_d(values),
            "mean_boot_ci95_low": mean_ci_low,
            "mean_boot_ci95_high": mean_ci_high,
            "median_boot_ci95_low": median_ci_low,
            "median_boot_ci95_high": median_ci_high,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p_onesided_greater": wilcoxon_p,
        }
        rows.append(row)

    test_df = pd.DataFrame(rows)

    paired_rows: list[dict[str, Any]] = []
    for model, model_means in pair_means.groupby("model"):
        base = model_means[model_means["condition"] == "useEffect"][
            ["pair_id", "mean_LD_stateform_pair"]
        ].rename(columns={"mean_LD_stateform_pair": "useEffect_pair_mean"})
        for condition in CONDITIONS:
            cond = model_means[model_means["condition"] == condition][
                ["pair_id", "mean_LD_stateform_pair"]
            ]
            merged = cond.merge(base, on="pair_id", how="inner")
            diff = (
                merged["mean_LD_stateform_pair"] - merged["useEffect_pair_mean"]
            ).to_numpy(dtype=float)
            if condition == "useEffect":
                paired_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "paired_vs_useEffect_n": len(diff),
                        "paired_vs_useEffect_mean_diff": 0.0,
                        "paired_vs_useEffect_t": math.nan,
                        "paired_vs_useEffect_p_twosided": math.nan,
                        "paired_vs_useEffect_cohend": math.nan,
                    }
                )
            else:
                t_stat, p_two = stats.ttest_1samp(diff, 0.0)
                paired_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "paired_vs_useEffect_n": len(diff),
                        "paired_vs_useEffect_mean_diff": float(diff.mean()),
                        "paired_vs_useEffect_t": float(t_stat),
                        "paired_vs_useEffect_p_twosided": float(p_two),
                        "paired_vs_useEffect_cohend": cohens_d(diff),
                    }
                )

    paired_df = pd.DataFrame(paired_rows)
    test_df = test_df.merge(paired_df, on=["model", "condition"], how="left")
    return pair_means, test_df


def make_wide(df: pd.DataFrame, pair_means: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["combo"] = (
        "sr="
        + work["state_role"]
        + "|do="
        + work["decl_order"]
        + "|bo="
        + work["body_order"]
    )
    pivot = work.pivot_table(
        index=["model", "condition", "pair_id"],
        columns="combo",
        values=[
            "D",
            "LD_stateform",
            "LD_reactive",
            "logit_dep",
            "logit_alt",
            "logit_useState_form",
            "logit_const_form",
            "logit_close",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{value}__{combo}" for value, combo in pivot.columns]
    pivot = pivot.reset_index()
    return pivot.merge(pair_means, on=["model", "condition", "pair_id"], how="left")


def regression(df: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm

    rows: list[dict[str, Any]] = []
    for model, group in df.groupby("model"):
        work = group.copy()
        work["pos_dep_first_code"] = np.where(work["pos_dep_first"], 1.0, -1.0)
        work["role_dep_useState_form_code"] = np.where(
            work["role_dep_useState_form"], 1.0, -1.0
        )
        work["decl_dep_first_code"] = np.where(work["decl_dep_first"], 1.0, -1.0)

        feature_cols = [
            "pos_dep_first_code",
            "role_dep_useState_form_code",
            "decl_dep_first_code",
        ]
        for condition in CONDITIONS[1:]:
            col = f"condition_{condition}"
            work[col] = (work["condition"] == condition).astype(float)
            feature_cols.append(col)
        for condition in CONDITIONS[1:]:
            col = f"role_dep_useState_form_x_{condition}"
            work[col] = (
                work["role_dep_useState_form_code"]
                * (work["condition"] == condition).astype(float)
            )
            feature_cols.append(col)

        x = sm.add_constant(work[feature_cols], has_constant="add")
        y = work["D"].astype(float)
        result = sm.OLS(y, x).fit(
            cov_type="cluster",
            cov_kwds={"groups": work["pair_id"]},
        )
        conf = result.conf_int()
        for term in result.params.index:
            rows.append(
                {
                    "model": model,
                    "method": "ols_cluster",
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


def write_validation_report(
    out_dir: Path,
    structural: dict[str, Any],
    availability: Availability,
    token_failures: list[dict[str, str]] | None = None,
    failed_pairs: set[str] | None = None,
    token_validation_status: str = "not run",
) -> None:
    token_failures = token_failures or []
    failed_pairs = failed_pairs or set()
    lines = [
        "# Reactivity Generalization Validation Report",
        "",
        "## Dataset",
        "",
        f"- rows: {structural['n_rows']}",
        f"- pairs: {structural['n_pairs']}",
        f"- conditions: {', '.join(structural['conditions'])}",
        f"- structural validation: {structural['structural_validation']}",
        "",
        "## Token/Model Availability",
        "",
    ]
    if availability.ok:
        lines.append("- local model stack/tokenizers: available")
    else:
        lines.append(f"- local model stack/tokenizers: unavailable")
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
            lines.append(
                f"- {failure['model']} {failure['row_id']}: {failure['reason']}"
            )
    lines.extend(
        [
            "",
            "## TODO",
            "",
            "Run the logit step in an environment with local Llama 3.2 1B and Gemma 3 1B-pt weights:",
            "",
            "```bash",
            "python3 reactivity_generalization/run_reactivity_generalization.py",
            "```",
        ]
    )
    (out_dir / "reactivity_generalization_validation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
    if df.empty:
        return ["(empty)"]
    sub = df[columns].head(max_rows).copy()
    return sub.to_markdown(index=False).splitlines()


def write_summary(
    out_dir: Path,
    structural: dict[str, Any],
    availability: Availability,
    logits_written: bool,
    pairlevel: pd.DataFrame | None = None,
    reg: pd.DataFrame | None = None,
    token_failures: list[dict[str, str]] | None = None,
    failed_pairs: set[str] | None = None,
    token_validation_status: str = "not run",
) -> None:
    token_failures = token_failures or []
    failed_pairs = failed_pairs or set()
    lines = [
        "# Reactivity Generalization Summary",
        "",
        "## Dataset",
        "",
        f"- rows: {structural['n_rows']}",
        f"- pairs: {structural['n_pairs']}",
        f"- conditions: {', '.join(structural['conditions'])}",
        f"- structural validation: {structural['structural_validation']}",
        f"- token validation: {token_validation_status}",
        f"- token-validation failures: {len(token_failures)}",
        f"- failed pairs dropped/flagged: {len(failed_pairs)}",
        "",
        "## Sign Convention",
        "",
        "`D = logit(dep) - logit(alt)`.",
        "",
        "`LD_stateform = (role_dep_useState_form ? +1 : -1) * D`.",
        "",
        "`LD_reactive` is emitted only as a backward-compatible alias for `LD_stateform`.",
        "",
    ]

    if not logits_written:
        lines.extend(
            [
                "## Status",
                "",
                "Stopped before logit measurement.",
                "",
                f"Reason: `{availability.reason}`",
                "",
                "TODO: rerun this script in the GPU/model environment with local weights.",
            ]
        )
    else:
        assert pairlevel is not None
        assert reg is not None
        lines.extend(
            [
                "## Stage A",
                "",
                "Pair-level mean of the eight cells per model/condition.",
                "",
            ]
        )
        lines.extend(
            markdown_table(
                pairlevel,
                [
                    "model",
                    "condition",
                    "n_pairs",
                    "pair_mean_mean",
                    "pair_mean_median",
                    "frac_pair_mean_pos",
                    "mean_boot_ci95_low",
                    "mean_boot_ci95_high",
                    "cohend",
                ],
                max_rows=20,
            )
        )
        key_terms = reg[
            reg["term"].str.contains("role_dep_useState_form", regex=False)
        ]
        lines.extend(
            [
                "",
                "## Stage B",
                "",
                "Regression dependent variable: `D = logit(dep) - logit(alt)`.",
                "",
            ]
        )
        lines.extend(
            markdown_table(
                key_terms,
                ["model", "term", "estimate", "std_err", "ci_low", "ci_high", "p_value"],
                max_rows=20,
            )
        )
        lines.extend(
            [
                "",
                "## Outputs",
                "",
                "- `reactivity_generalization_logits_3tokens.csv`",
                "- `reactivity_generalization_wide_per_sample.csv`",
                "- `reactivity_generalization_pairlevel_test.csv`",
                "- `reactivity_generalization_regression.csv`",
            ]
        )

    (out_dir / "reactivity_generalization_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_outputs(
    out_dir: Path,
    df: pd.DataFrame,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logits_path = out_dir / "reactivity_generalization_logits_3tokens.csv"
    wide_path = out_dir / "reactivity_generalization_wide_per_sample.csv"
    pairlevel_path = out_dir / "reactivity_generalization_pairlevel_test.csv"
    regression_path = out_dir / "reactivity_generalization_regression.csv"

    pair_means, pairlevel = stage_a_pairlevel(df, bootstrap_samples)
    wide = make_wide(df, pair_means)
    reg = regression(df)

    df.to_csv(logits_path, index=False)
    wide.to_csv(wide_path, index=False)
    pairlevel.to_csv(pairlevel_path, index=False)
    reg.to_csv(regression_path, index=False)
    return pairlevel, reg


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(dataset_path)
    structural = validate_structure(rows, args.expected_pairs)

    availability = Availability(True)
    token_failures: list[dict[str, str]] = []
    failed_pairs: set[str] = set()

    if args.skip_logits:
        availability = Availability(False, "--skip-logits was set")
        write_validation_report(
            out_dir,
            structural,
            availability,
            token_validation_status="not run (--skip-logits)",
        )
        write_summary(
            out_dir,
            structural,
            availability,
            logits_written=False,
            token_validation_status="not run (--skip-logits)",
        )
        print("Structural validation passed; skipped logit measurement.")
        return 0

    availability, torch, auto_model, auto_tokenizer = import_model_stack()
    if not availability.ok:
        write_validation_report(
            out_dir,
            structural,
            availability,
            token_validation_status="not run (model stack unavailable)",
        )
        write_summary(
            out_dir,
            structural,
            availability,
            logits_written=False,
            token_validation_status="not run (model stack unavailable)",
        )
        print(f"Structural validation passed; stopped before logits: {availability.reason}")
        return 0

    local_files_only = not args.allow_downloads
    tok_availability, tokenizers = load_tokenizers(auto_tokenizer, local_files_only)
    if not tok_availability.ok:
        write_validation_report(
            out_dir,
            structural,
            tok_availability,
            token_validation_status="not run (tokenizers unavailable)",
        )
        write_summary(
            out_dir,
            structural,
            tok_availability,
            logits_written=False,
            token_validation_status="not run (tokenizers unavailable)",
        )
        print(f"Structural validation passed; stopped before logits: {tok_availability.reason}")
        return 0

    token_info, token_failures, failed_pairs = validate_tokens_for_models(rows, tokenizers)
    valid_rows = [row for row in rows if row["pair_id"] not in failed_pairs]
    if not valid_rows:
        availability = Availability(False, "all pairs failed tokenizer validation")
        write_validation_report(
            out_dir,
            structural,
            availability,
            token_failures,
            failed_pairs,
            token_validation_status="failed",
        )
        write_summary(
            out_dir,
            structural,
            availability,
            logits_written=False,
            token_failures=token_failures,
            failed_pairs=failed_pairs,
            token_validation_status="failed",
        )
        print("Tokenizer validation failed for all pairs; stopped before logits.")
        return 0

    device = resolve_device(torch, args.device)
    try:
        df = measure_logits(
            valid_rows,
            tokenizers,
            token_info,
            torch,
            auto_model,
            device,
            local_files_only,
        )
    except Exception as exc:  # pragma: no cover - depends on local env
        availability = Availability(False, f"model weights unavailable or load failed: {exc}")
        write_validation_report(
            out_dir,
            structural,
            availability,
            token_failures,
            failed_pairs,
            token_validation_status="passed"
            if not token_failures
            else "passed after dropping flagged pairs",
        )
        write_summary(
            out_dir,
            structural,
            availability,
            logits_written=False,
            token_failures=token_failures,
            failed_pairs=failed_pairs,
            token_validation_status="passed"
            if not token_failures
            else "passed after dropping flagged pairs",
        )
        print(f"Stopped before writing logits: {availability.reason}")
        return 0

    pairlevel, reg = write_outputs(out_dir, df, args.bootstrap_samples)
    write_validation_report(
        out_dir,
        structural,
        Availability(True),
        token_failures,
        failed_pairs,
        token_validation_status="passed"
        if not token_failures
        else "passed after dropping flagged pairs",
    )
    write_summary(
        out_dir,
        structural,
        Availability(True),
        logits_written=True,
        pairlevel=pairlevel,
        reg=reg,
        token_failures=token_failures,
        failed_pairs=failed_pairs,
        token_validation_status="passed"
        if not token_failures
        else "passed after dropping flagged pairs",
    )
    print(f"Wrote generalization outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
