# Revision-round analyses (2026-10)

Supplementary analyses added during the first revision round. Each folder holds
the evaluation/report scripts and the result note (Korean, verbatim lab record;
the English citation sentence is at the end of each note). Raw per-instance
JSONL outputs are not distributed (see the data statement in the repository
README); attribution vectors are in `../attributions/`.

All scripts assume the toolkit layout of this repository (`toolkit/kodocxai`)
plus the fine-tuned checkpoints, which are reproducible via
`kodx_train.py` (seeds 1-5) as described in the repository README.

| Folder | One-line description | Reproduce |
|---|---|---|
| `P11_rollout_span` | Attention-rollout with span-level targets (mean/sum over span tokens), 3 seeds | `python eval_rollout_span.py --seed {1,2,3}` then `python report_p11.py` |
| `P12_alternative_perturbations` | Deletion / random-substitution perturbations vs [MASK], fixed 100-document sample (`sample100.json`) | `python make_sample_p12.py` (regenerates the sample; `verify_sample_p12.py` checks it), `python eval_perturb.py --seed {1,2,3}`, `python report_p12.py` |
| `P13_reference_dilation` | Sensitivity of plausibility to reference-set dilation (10/25/50% box expansion) | `python report_p13.py` (recomputes from phi vectors) |
| `P14_seed_paired` | Seed-paired differences (IG-random, IG-rollout) with document-cluster bootstrap, 3 seeds | `python report_p14.py` |
| `P15_additional_attributions` | Gradient×Input and leave-one-out occlusion added to the method set, 3 seeds | `python eval_newattr.py --seed {1,2,3}` then `python report_p15.py` |
| `P16_additional_seeds` | Two additional training seeds (4, 5); five-seed extension of the paired-difference analysis | `run_train_45.ps1`, evaluation via `kodx_eval.py --skip-naopc`, `python backfill_phi.py --seed {4,5}`, `python report_p16.py` |
| `P17_lilt` | Alternative architecture: LiLT (no visual features; layout flow of lilt-infoxlm-base + XLM-R text encoder), 1 seed | `python train_lilt.py --xlmr-init --no-checkpointing`, `python run_eval_lilt.py kie` / `attrib`, `python report_p17.py` |
