# Reactivity Generalization Validation Report

## Dataset

- rows: 1800
- pairs: 45
- conditions: useEffect, useLayoutEffect, alias, alias_ctrl, subscribe
- structural validation: passed

## Token/Model Availability

- local model stack/tokenizers: unavailable
- reason: `model stack unavailable: No module named 'torch'`
- token validation: not run (model stack unavailable)
- token-validation failures: 0
- failed pairs flagged for dropping: 0

## TODO

Run the logit step in an environment with local Llama 3.2 1B and Gemma 3 1B-pt weights:

```bash
python3 reactivity_generalization/run_reactivity_generalization.py
```
