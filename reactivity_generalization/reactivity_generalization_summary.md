# Reactivity Generalization Summary

## Dataset

- rows: 1800
- pairs: 45
- conditions: useEffect, useLayoutEffect, alias, alias_ctrl, subscribe
- structural validation: passed
- token validation: not run (model stack unavailable)
- token-validation failures: 0
- failed pairs dropped/flagged: 0

## Sign Convention

`D = logit(dep) - logit(alt)`.

`LD_stateform = (role_dep_useState_form ? +1 : -1) * D`.

`LD_reactive` is emitted only as a backward-compatible alias for `LD_stateform`.

## Status

Stopped before logit measurement.

Reason: `model stack unavailable: No module named 'torch'`

TODO: rerun this script in the GPU/model environment with local weights.
