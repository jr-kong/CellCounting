Activate venv
.\.venv\Scripts\Activate

Preprocess data
python data_preprocess.py --output-dir processed --photometric-samples 2

Key flags
--image-dir: Directory containing the raw w1 images (default w1_images).
--mask-dir: Directory containing the binary masks (default w1_mask).
--output-dir: Destination root for metadata and sample NPZ files (default processed).
--sample-dir: Subdirectory under output-dir for per-sample NPZs (default samples).
--metadata-name: Filename for the CSV metadata (default metadata.csv).
--rotations: Right-angle rotations to apply, e.g. 0 90 180 270.
--photometric-samples: Number of random photometric variants per rotation.
--target-size: Optional pad size if you want to override automatic sizing.
--precision: float32 or float16 output tensors (default float32).

Train CNN
python train_cnn.py --metadata processed/metadata.csv --root processed --epochs 20 --val-split 0.1 --test-split 0.1 --experiment-root experiments --run-name cnn_1

Training artifacts
- Metrics tracked: per-epoch train MSE and (when enabled) validation MSE/MAE.
- Each run writes config.json, metrics.csv, test_metrics.json (when a test split is used), model.pt, and test_scatter.png into experiments/<run-name-or-timestamp>/. Training shows tqdm progress bars for train/val epochs.
