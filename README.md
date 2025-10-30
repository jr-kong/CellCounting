**Setup**
- Activate virtual environment: `.\.venv\Scripts\Activate`
- Install dependencies: `pip install -r requirements.txt`

**Preprocess Data**
- Run: `python data_preprocess.py --output-dir processed --photometric-samples 2`
- Flags:
  - `--image-dir`: folder with raw w1 images (default `w1_images`)
  - `--mask-dir`: folder with matching masks (default `w1_mask`)
  - `--output-dir`: root for metadata and NPZ exports (default `processed`)
  - `--sample-dir`: subfolder under output-dir for sample NPZs (default `samples`)
  - `--metadata-name`: CSV metadata filename (default `metadata.csv`)
  - `--rotations`: right-angle rotations to apply, e.g. `0 90 180 270`
  - `--photometric-samples`: random photometric variants per rotation (default `0`)
  - `--brightness-shift`: brightness jitter range in 0–1 space (default `0.1`)
  - `--contrast-range`: contrast multiplier range (default `0.1`)
  - `--gamma-range`: gamma jitter range (default `0.1`)
  - `--noise-std`: additive Gaussian noise std (default `0.01`)
  - `--target-size`: square padding size; inferred when omitted
  - `--precision`: stored tensor dtype (`float32` or `float16`)
  - `--seed`: RNG seed for augmentations (default `42`)

**Train CNN**
- Run: `python /content/CellCounting/train_cnn.py --run-name cnn_1`
- Flags:
  - `--metadata`: path to metadata CSV
  - `--root`: root containing the per-sample NPZ files
  - `--batch-size`: mini-batch size (default `4`)
  - `--epochs`: number of epochs (default `10`)
  - `--learning-rate`: Adam learning rate (default `1e-3`)
  - `--val-split`: validation fraction (default `0.1`)
  - `--test-split`: test fraction (default `0.0`)
  - `--num-workers`: dataloader worker count (default `0`)
  - `--seed`: random seed for shuffling and weight init
  - `--device`: training device (`cuda` or `cpu`)
  - `--experiment-root`: folder for run artifacts
  - `--run-name`: optional explicit run identifier

**Outputs**
- Preprocessing writes `metadata.csv` plus per-sample NPZ files and logs in `processed/`
- Training saves `config.json`, `metrics.csv`, optional `test_metrics.json`, `model.pt`, and `test_scatter.png` under each experiment run
- Progress bars (tqdm) appear for train/val epochs; disable by redirecting output or editing the script if desired
