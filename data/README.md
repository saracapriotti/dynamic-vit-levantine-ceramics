# Dataset

The petrographic ceramics dataset used in the paper is **not public** (available on
request — contact the authors). This folder is where it goes locally; nothing here
is tracked by git except this file.

## Expected layout

```
data/
├── dataset.csv     # ';'-separated, latin-1 encoded
│                   #   column 0 -> image path relative to DATASET/
│                   #   column 1 -> class label (1-20, shifted to 0-19 at load time)
└── DATASET/        # flat folder of images, referenced by dataset.csv
```

20 classes (petrographic fabrics). The 70/15/15 train/val/test split is fixed by
`SEED = 5` (see `src/baselines/config.py` and the notebooks' configuration cells) and
is identical across every run and every model, so results are comparable.

## Preparing the images

The source images are typically distributed as per-class zip archives. Extract them
all into a single flat folder:

```bash
cd data
mkdir -p DATASET
for z in /path/to/archives/*.zip; do unzip -j "$z" -d DATASET/; done
```

Then point the code at this folder:

- **Notebooks** (`notebooks/01_training.ipynb`, `notebooks/02_xai.ipynb`): set `DATA_DIR`
  in the configuration cell to the absolute path of this `data/` folder (or, on Colab,
  to the corresponding Google Drive path).
- **Dynamic ViT CLI** (`src/dynamicvit/`): set `DATA_DIR` in `config.py` the same way.
- **Baselines** (`src/baselines/`): set `DATASET_DIR` in `config.py` the same way.

> Compressed archives total ~22 GB, extracted ~39 GB. Plan for ~61 GB of free disk space.
