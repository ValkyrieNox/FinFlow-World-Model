# Heston V3 Data Generation

Generate the default V3 dataset:

```bash
python3 scripts/generate_heston_data.py --output data/heston_v3
```

For a small smoke run:

```bash
python3 scripts/generate_heston_data.py --output /tmp/finflow_heston_smoke --n-train 4 --n-val 2 --n-test 2 --steps 5
```

## Outputs

Each split writes:

- `{split}.npz`: full simulated paths with `s_paths`, `v_paths`, and `log_returns`
- `{split}_transitions.npz`: flattened one-step transition samples
- `metadata.json`: Heston parameters, split sizes, transition alignment, and train-set normalization stats

Transition alignment is:

- condition: `(v_t, r_{t-1})`, with `r_{-1}=0`
- target: `(v_{t+1}, r_t)`

The transition file also includes `log_v_t` and `log_v_next`, which are the intended variance features for model training.

