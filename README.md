# De-identification System Evaluation Code

Evaluation and scoring code for the study *"Head-to-head evaluation of commercial and
open-source de-identification systems for Korean clinical records."*

This repository contains the evaluation harness, statistical-comparison scripts, and the
open-source baseline (fine-tuning and inference) code used in the study. It does **not**
contain any clinical data, gold-standard annotations, or vendor software.

## Important: what is and is not included

- **Included:** scoring harness, label-harmonization logic, statistical tests, and scripts
  to run the open-source baselines (GLiNER, KoELECTRA, KLUE-BERT).
- **Not included, and cannot be shared:**
  - The clinical corpus and the annotated gold standard (they contain protected health
    information and are governed by the institutional review board approvals).
  - The two commercial de-identification programs (Program A and Program B), which are
    proprietary products of their respective vendors.

Vendor names are anonymized as **Program A** and **Program B** throughout, consistent with
the manuscript.

## Contents

| File | Purpose |
|------|---------|
| `gold_loader.py` | Load Label-Studio-format gold annotations from a directory of JSON files and merge them into per-document PHI spans. |
| `deid_eval.py` | Strict and lenient span-level scoring (micro precision, recall, F1; zero-missed-PHI rate). Shared by all systems. |
| `run_gliner_zeroshot.py` | Run the multilingual GLiNER model without clinical fine-tuning; optional decision-threshold sweep. |
| `run_klue_bert.py` | Fine-tune KLUE-BERT for PHI token classification with an 80/20 document-level split. |
| `run_koelectra_ablation.py` | Evaluate KoELECTRA before and after clinical fine-tuning on the same held-out test documents. |
| `run_ab_stats.py` | Reproduce the Program A vs Program B statistical comparison (bootstrap CIs, Wilcoxon signed-rank, McNemar) from per-document score tables. |

## Requirements

- Python 3.11
- `numpy`, `scipy`, `openpyxl` (statistics and I/O)
- `torch`, `transformers`, `gliner` (open-source baselines only)

```
pip install numpy scipy openpyxl torch transformers gliner
```

## Data format

The scripts expect **your own** gold annotations and system outputs; no data is distributed here.

- Gold annotations: a directory of Label-Studio export JSON files, passed via `--gold_dir`.
- Per-document vendor scores (for `run_ab_stats.py`): an `.xlsx` file with a per-document
  sheet of strict TP/FP/FN counts, one file per vendor.

## Reproducing the statistical comparison

```
python run_ab_stats.py \
  --a_xlsx vendorA_admission/prelim_scores.xlsx \
  --b_xlsx vendorB_admission/prelim_scores.xlsx \
  --record admission --out stats_admission.json
```

`--record discharge` with the discharge files reproduces the discharge comparison.

## Scoring definitions

- **Strict match:** document, start offset, end offset, and label all match a gold span.
- **Lenient match:** document and character offsets match (label ignored).
- **Micro P/R/F1:** pooled over all labels and documents.
- **zero-missed-PHI rate:** proportion of documents with no missed gold span
  (document-level false negatives = 0).
- **Detection accuracy** (external-site metric): TP/(TP+FP+FN), a Jaccard-type index with
  no true negatives; related to F1 by F1 = 2 x accuracy / (1 + accuracy).

## Citation

Please cite the associated article (details to be added on publication).

## License

MIT License (code only; no data).
