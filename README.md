# KoDocXAI

**KoDocXAI** is a quantitative evaluation framework, benchmark, and toolkit for
explainability in key information extraction (KIE) from Korean documents. It
accompanies the paper:

> Park, M.; Jung, H.-Y.; Seo, S. *KoDocXAI: A Quantitative Evaluation Framework
> and Benchmark for Explainability in Key Information Extraction from Korean
> Administrative Form Documents.* Applied Sciences (revised version under review, Manuscript ID applsci-4544343).
> `[citation will be updated upon publication]`

The framework derives a value-region rationale reference from existing KIE
bounding-box annotations and scores value-region plausibility (whether an
explanation localises the extracted value), re-aggregates subword attributions
to morphemes and eojeols, and measures faithfulness by modality-wise
perturbation with pretraining-consistent [MASK] replacement while protecting
the ground-truth span. See the paper for the full protocol and results.

Archived at Zenodo: DOI [10.5281/zenodo.21964543](https://doi.org/10.5281/zenodo.21964543).
The annotation layer is also mirrored at
[Hugging Face](https://huggingface.co/datasets/kodocxai/kodocxai).

## Structure and licensing

Licensing is per directory - the brand is one, the licenses are two.

| Path | Contents | License |
|---|---|---|
| `annotations/` | **KoDocXAI benchmark v1.0** public annotation layer - document identifiers, field types, status labels, annotation-ID index spans, label-file paths, integrity hashes. Format: `annotations/FORMAT.md`. **Contains no AI-Hub source content** (no images, no transcribed text, no coordinates) | CC BY 4.0 |
| `toolkit/` | `kodocxai` Python package - the full evaluation pipeline (fine-tuning, IG / attention-rollout attribution, unit re-aggregation subword→morpheme→eojeol, plausibility and faithfulness scoring, modality-wise perturbation) | MIT |
| `annotation-tool/` | Web-based annotation tool used to build the benchmark (Python stdlib only) | MIT |
| `scripts/` | `reassemble.py` (join the annotation layer with your own AI-Hub download; deterministic, manifest-based) and `verify.py` (integrity check) | MIT |
| `guidelines/` | Annotation guideline v1.4, full text (Korean; an English version is provided in the paper's Supplementary Material S1) | CC BY 4.0 |
| `rubric/` | Qualitative-layer rubric and application protocol (Korean; English version in Supplementary Material S2) | CC BY 4.0 |
| `analyses/` | Scripts and result tables of the additional analyses of the revised version (span-level rollout target, alternative text perturbations with the fixed 100-document sample list, matched perturbation budgets, dilated reference regions, seed-paired differences, two further attribution methods, two further training seeds, a second LiLT backbone); see `analyses/README.md` | MIT |
| `attributions/` | Per-instance attribution scores (`phi_*.jsonl`: document identifier, field, method, and a 512-value score vector per instance) for the fine-tuned seeds and the LiLT backbone, so that the plausibility and faithfulness tables can be recomputed without re-running attribution | CC BY 4.0 |

## What is not included, and why

The AI-Hub source dataset (공공행정문서 OCR / Public Administrative Documents
OCR, dataSetSn=88) may not be provided or redistributed to third parties under
its terms of use (terms verified on 15 August 2026). The original images,
transcribed text, and coordinates are therefore **not** part of this
repository, and every user - including co-researchers - must apply for and
download the source dataset individually. Applications are currently limited
to residents of the Republic of Korea, so full reconstruction is domestically
restricted (see the paper, Section 5.5).

## Reconstructing the benchmark

1. Apply for and download 공공행정문서 OCR (dataSetSn=88) at
   [AI-Hub](https://www.aihub.or.kr). `Training/[라벨]train.zip` is required;
   the document images used by the image modality come from
   `Training/[원천]train*.zip`.
2. Reassemble:

   ```
   python scripts/reassemble.py --label-zip path/to/[라벨]train.zip \
          [--src-zip path/to/[원천]train1.zip ...] [--out reassembled]
   ```

3. Integrity check only (nothing written):

   ```
   python scripts/reassemble.py --label-zip path/to/[라벨]train.zip --verify-only
   python scripts/verify.py
   ```

Reassembly is deterministic and manifest-based; every span is verified against
the SHA-256 hashes recorded in `annotations/manifest_public.json`.

## Reproducing the paper's tables

`toolkit/kodocxai/` holds the training (`kodx_train.py`), evaluation
(`kodx_eval.py`) and reporting (`report_kodx.py`) pipeline of the main
experiments. `analyses/` holds the scripts of the additional analyses reported
in Supplementary Document S4 of the revised version, each with its result
table; the attribution vectors they consume are in `attributions/`, so the
tables can be recomputed on CPU from the released files once the benchmark has
been reassembled.

## Note on paths

The scripts are released exactly as they were run, so some default arguments
point to the authors' local layout (a drive letter and a `_pilot/` folder) and
to the primary annotation file under its working name. Every such path can be
overridden on the command line (`--docs`, `--annotations`, `--images`,
`--model`, `--out`); after reassembly, pass the reassembled `docs.jsonl` and
`annotations.jsonl` explicitly. The defaults are kept verbatim rather than
edited so that the released code is byte-identical to the code that produced
the reported numbers.

## Note on naming

The internal working name of this project was KoDX-Eval; the public name is
KoDocXAI. Module file names (`kodx_*.py`) and the fixed randomization seed
strings `"kodx-eval-v1"` and `"kodx-naopc-v1"` are retained verbatim because
changing them would break the reproducibility of the released sample selection
and presentation order.

## Attribution (required by the source-data provider)

This research (paper) used datasets from 'The Open AI Dataset Project (AI-Hub,
S. Korea)'. All data information can be accessed through 'AI-Hub
(www.aihub.or.kr)'.

## Citing

Until the paper is published, please cite the Zenodo archive
(DOI 10.5281/zenodo.21964543). A BibTeX entry for the paper will be added upon
publication.
