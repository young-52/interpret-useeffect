# Reactivity Ablation Summary

## Dataset

- rows: 2520 (C: 1800, B: 720)
- pairs: 45
- structural validation: passed
- token validation: not run (model stack unavailable)

## Sign Convention

`D = logit(dep) - logit(alt)`.

`LD_stateform = (role_dep_useState_form ? +1 : -1) * D`.

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

## Scope note (Ablation B, optional)

Headline framing is not fixed in advance. If the effect holds in `useEffect` and
`subscribe` but drops in `plain_array`/`return_array`, that reads as gated by
callback-array context. If it holds across all four, that reads as a broader
array-completion prior not specific to callback wrapping. The phrase "beyond dependency
arrays" is licensed only by the second pattern. Until resolved, all claims here are scoped
to callback-array completion, not dependency-array completion specifically.

## Status

Stopped before logit measurement.

Reason: `model stack unavailable: No module named 'torch'`

TODO: rerun this script in the GPU/model environment with local weights.
