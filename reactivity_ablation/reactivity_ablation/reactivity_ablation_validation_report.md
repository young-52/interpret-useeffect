# Reactivity Ablation Validation Report

## Dataset

- rows: 2520 (C: 1800, B: 720)
- pairs: 45
- ablation C contexts: useEffect, subscribe
- ablation B contexts: useEffect, subscribe, plain_array, return_array
- structural validation: passed

## Token/Model Availability

- local model stack/tokenizers: unavailable
- reason: `model stack unavailable: No module named 'torch'`
- token validation: not run (model stack unavailable)
- token-validation failures: 0
- failed pairs flagged for dropping: 0

## TODO

Run the logit step in an environment with local Llama 3.2 1B and Gemma 3 1B-pt weights
(run from inside this directory so the relative dataset path resolves):

```bash
cd reactivity_ablation && python3 run_reactivity_ablation.py
```
