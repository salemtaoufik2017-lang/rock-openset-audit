# rock-openset-audit

Code and results for the paper *Open-Set and Hierarchical Lithology Recognition from Rock Photographs for Detecting Unseen Rock Types across Two Public Image Collections* (International Journal of Intelligent Engineering and Systems).

The code merges two public rock-photograph collections, removes duplicates and augmented copies, and evaluates closed-set, hierarchical and open-set recognition with group-aware splits.

## Data

The images are not stored here. Download them from the original sources:

- **D1**: Kaggle, "Rock Classification Dataset", Version 1 (24 April 2021), DOI 10.34740/kaggle/ds/1293628: https://www.kaggle.com/datasets/salmaneunus/rock-classification. Uploaded by Salman Ibne Eunus and compiled by members of the BRAC University Mars Rover team from images scraped from Google Images and other machine-learning repositories. No licence terms are stated, so the images are not redistributed here.
- **D2**: Roboflow Universe, project `rock-clasfication`, version 1 (MIT licence): https://universe.roboflow.com/william-cwsr-8jizi/rock-clasfication

## How to reproduce

1. Open `Revision_Experiments.ipynb` in Google Colab and select an A100 GPU (Runtime > Change runtime type).
2. Run all cells. The notebook downloads both datasets and runs every step; set a free Roboflow API key in cell 1 if the Kaggle copy of D2 is not available.
3. Results are written to `/content/rock_revision_out` and packed into `rock_revision_results_final.zip`.

The same steps can be run from the command line:

```
python rock_revision.py audit      # leakage audit, units, folds
python rock_revision.py features   # descriptor and frozen embeddings
python rock_revision.py leakage    # Table 4 (protocol comparison)
python rock_revision.py probe      # collection probes
python rock_revision.py finetune   # five-fold, O1/O2 and P3b fine-tuning
python rock_revision.py analyse    # score selection, Holm, bootstrap
```

Edit `CONFIG` at the top of `rock_revision.py` to set data paths. A full run takes 2 to 4 hours on one A100.

## Contents

| Path | What it holds |
|---|---|
| `rock_revision.py` | All code: hashing and grouping, descriptor, fine-tuning, rejection scores, statistics |
| `Revision_Experiments.ipynb` | Colab notebook that runs the full pipeline |
| `data/units_manifest.csv` | The deduplicated units (MD5, collection, type, leakage group) |
| `data/folds.csv` | Fold index of every unit for seeds 0, 1 and 2 |
| `results/audit/` | Image-to-group mapping, conflict groups, component sizes, D2 export counts, inspected dHash pairs |
| `results/leakage/` | Table 4 rows (random, source-grouped and unit-level splits) |
| `results/probe/` | Collection probes on all units and on Basalt only |
| `results/results/` | Five-fold, open-set and P3b results, validation-based score selection, Holm-adjusted tests, bootstrap intervals, per-unit predictions, configuration and software versions |
| `results/ft_curves/` | Training and validation loss per epoch for every fine-tuning run |

## What is not included

- Model checkpoints were not kept. Fine-tuned results are reproduced by retraining with `rock_revision.py` (2 to 4 GPU hours on one A100).
- Open-set rejection scores are given per run and per score (`results/results/osr_ft_all_scores.csv`, `p3b_ft_all.csv`), not per image.
- The 3,521-unit manifest, folds and runs of the first submission came from an earlier notebook and are not part of this repository. All analyses added in the revision use the 3,545 rebuilt units in `data/`.
- Class labels are those of the dataset compilers and were not re-checked by a geologist.

## Settings

All parameters are listed in Table 3 of the paper and in `results/results/config_used.json`. Software versions are in `results/results/software_versions.json`.

## Citation

Please cite the paper if you use this code. The full reference will be added after publication.
