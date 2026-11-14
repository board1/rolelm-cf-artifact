# Counterfactual RoleRM adapter

Anonymous review artifact for a local counterfactual-DPO adaptation of `HeAAAAA/RoleRM`.

## Identity

- Base repository: `HeAAAAA/RoleRM`
- Base revision: `52ae4bf5ba2616ec48834f8467ee1af88443161a`
- Adapter weight SHA-256: `8ace45df92e2a2c5093ca6ee7733814e85dc44053ddd3513d7f2fb1af80e04b5`
- PEFT LoRA rank16, alpha32, q/v/o targets, dropout0.05.
- Saved training settings:2500 preference pairs,2 epochs, DPO beta0.5, learning rate1e-4.

The adapter weights are unchanged. The adapter config replaces the original machine-specific base path with the pinned public base identifier. `MANIFEST.json` identifies the distributed files. This package reconstructs the saved adaptation; it does not include the preference training dataset and is not a complete re-training release.

## Reconstruction

Install the dependencies, then run:

```bash
python reconstruct.py --output-dir merged_model
```

The base is approximately32GB in float32; allow sufficient CPU memory and disk space. The script merges in float32 and exports bfloat16. `export_fingerprints.json` identifies the original export. Serialized file hashes can differ with shard layout or metadata; `merge_audit.json` records the existing tensor-level equality check against the original saved export. Reconstruction has not been re-executed as part of packaging.

## Evaluation identity

The additional Llama configuration used the merged adaptation for its training reward and the standard unadapted RoleRM for saved held-out evaluation. It is an additional policy–reward configuration, not a controlled policy-backbone-only comparison.

## Upstream terms

This is a derivative adapter, not a redistribution of the full upstream base. Consult the base repository and its underlying model terms. No new license over third-party materials is asserted here.
