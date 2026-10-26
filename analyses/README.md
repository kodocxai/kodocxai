# Additional analyses (2026-09/10)

Supplementary analyses added after the initial submission. Each folder holds
the evaluation/report scripts and the result note (Korean, verbatim lab
record; the notes of the revision-round analyses P11-P17 end with an English
citation sentence). Per-instance JSONL outputs are distributed only for the
folders that list them in the table below; those files contain document
identifiers, field names, method names, scores and token/eojeol indices only
(no document text and no coordinates, see the data statement in the
repository README). Attribution vectors are in `../attributions/`.

All scripts assume the toolkit layout of this repository (`toolkit/kodocxai`)
plus the fine-tuned checkpoints, which are reproducible via
`kodx_train.py` (seeds 1-5) as described in the repository README. Some
scripts keep the authors' local paths as defaults or as fixed inputs (see
"Note on paths" in the repository README); the "Reproduce" column lists the
arguments to pass explicitly after reassembly.

| Folder | One-line description | Reproduce |
|---|---|---|
| `P11_rollout_span` | Attention-rollout with span-level targets (mean/sum over span tokens), 3 seeds | `python eval_rollout_span.py --seed {1,2,3}` then `python report_p11.py` |
| `P12_alternative_perturbations` | Deletion / random-substitution perturbations vs [MASK], fixed 100-document sample (`sample100.json`) | `python make_sample_p12.py` (regenerates the sample; `verify_sample_p12.py` checks it), `python eval_perturb.py --seed {1,2,3}`, `python report_p12.py` |
| `P13_reference_dilation` | Sensitivity of plausibility to reference-set dilation (10/25/50% box expansion) | `python report_p13.py` (recomputes from phi vectors) |
| `P14_seed_paired` | Seed-paired differences (IG-random, IG-rollout) with document-cluster bootstrap, 3 seeds | `python report_p14.py` |
| `P15_additional_attributions` | Gradient×Input and leave-one-out occlusion added to the method set, 3 seeds | `python eval_newattr.py --seed {1,2,3}` then `python report_p15.py` |
| `P16_additional_seeds` | Two additional training seeds (4, 5); five-seed extension of the paired-difference analysis | `run_train_45.ps1`, evaluation via `kodx_eval.py --skip-naopc`, `python backfill_phi.py --seed {4,5}`, `python report_p16.py` |
| `P17_lilt` | Alternative architecture: LiLT (no visual features; layout flow of lilt-infoxlm-base + XLM-R text encoder), 1 seed | `python train_lilt.py --xlmr-init --no-checkpointing`, `python run_eval_lilt.py kie` / `attrib`, `python report_p17.py` |
| `P3_shortcut_baselines` | Shortcut baselines for plausibility: position prior (training-split value-span centres, `prior_train.json`) and label-nearest baseline, 3 seeds; per-instance outputs `P3_seed{1,2,3}.jsonl`, anchor-not-found lists `no_anchor_seed{N}.json` | `python eval_shortcut.py --model runs/seed{N}/last --docs docs.jsonl --images <images224> --annotations annotations.jsonl --train <train.jsonl> --out P3_seed{N}.jsonl --seed {N}` then `python report_p3.py` (reads the three JSONL files next to it and the main-evaluation records `seed{N}_v2.jsonl` from a fixed `_exp/results` path, see "Note on paths") |
| `P5_outcome_split` | Separation of explanation targets by prediction outcome (correct / wrong-partial / wrong-disjoint / missed), 3 seeds; per-instance predicted spans `P5_pred_seed{1,2,3}.jsonl` | `python dump_pred.py --model runs/seed{N}/last --phi ../../attributions/phi_seed{N}.jsonl --docs docs.jsonl --images <images224> --annotations annotations.jsonl --out P5_pred_seed{N}.jsonl` then `python report_p5.py` (fixed `_pilot`/`_exp` input paths, see "Note on paths") |
| `P18_layout_rerun` | Layout-perturbation columns of Table 7 recomputed (2026-10-26) with per-eojeol jitter and other-field coordinate swap (`jitter_boxes_by_word`, `swap_partner_tokens` in `kodx_attrib.py`), 3 seeds; per-instance outputs `layout_v3_seed{1,2,3}.jsonl`, table `layout_v3_table.json` | `python eval_layout_v3.py --model runs/seed{N}/last --docs docs.jsonl --images <images224> --annotations annotations.jsonl --out results/layout_v3_seed{N}.jsonl --seed {N}` (optional `--ref results/seed{N}_v2.jsonl` for the comparison columns) then `python report_layout_v3.py` (expects the three JSONL files under `results/` next to the script; move the distributed files there first). `split_stamp_v3.py` reads fixed `_pilot`/`_exp/results` paths (see "Note on paths") |
