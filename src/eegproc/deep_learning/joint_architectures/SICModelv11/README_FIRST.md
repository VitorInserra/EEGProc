# SIC MTLFuseNet-style 3D-CNN update

SIC builder API version 17 replaces the within-window temporal BiLSTM with a
spatio-temporal 3D-CNN while retaining the trial-level BiGRU classifier:

```text
channel-major 3-band EEG, one-second windows (128, 42)
    -> parallel GCN-GRU and MTLFuseNet-style 3D-CNN encoders
    -> direct per-window feature concatenation
    -> ordered full-trial BiGRU sequence
    -> one VariationalClassifier logits head
```

The loader bandpass-filters each complete trial into theta/alpha/beta before
windowing, producing the required channel-major `(128, 42)` inputs. The 3D-CNN
sums those bands into MTLFuseNet's raw-like 4--30 Hz spatio-temporal signal and
restores each timestep's 14 Emotiv electrodes to the same 9x9 scalp layout.
The input to `Conv3D` is `(time, grid-row, grid-column, 1)`. Convolutions span
time and both scalp axes, and global averaging produces one
`cnn3d_filters[-1]` embedding per one-second window. The graph branch likewise
computes window-level differential entropy before its GCN and spectral GRU.
The trial BiGRU therefore receives 60 ordered window embeddings instead of
7,680 raw-timestep embeddings for a standard DREAMER trial.

The default branch is:

```json
{
  "cnn3d_filters": {"fixed": [32, 64, 128]},
  "cnn3d_temporal_kernel_size": 7,
  "cnn3d_spatial_kernel_size": 3,
  "cnn3d_spatial_pool_sizes": {"fixed": [2, 2, 1]},
  "cnn3d_dropout": 0.2,
  "cnn3d_grid_size": 9,
  "use_cnn3d_branch": true
}
```

With the standard GCN-GRU width of 384, direct fusion is 384 + 128 = 512
features per timestep. The classifier still uses the tapered two-layer BiGRU
`[128, 64]`, and the subject adversary still receives its final trial state.

When reconstruction is enabled, the 3D-CNN feature sequence has its own
independent decoder. It is never decoded through the graph branch or a fused
representation. Calibration freezes the 3D-CNN backbone; unfreeze level 2
continues to mean trial BiGRU plus VC head.

## Longleaf order

The Conv3D branch processes every window in a trial batch. The launchers use a
memory-safe source batch of 8 trials for ERM and V-REx; increase it only after
measuring GPU memory. MLDG defaults to two trials per subject so arousal folds
with only one available trial in a subject/class pool remain valid.

Before the full experiment, submit the five classification-first diagnostics
from the project root:

```bash
bash src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/submit_sic_classification_diagnostics.sh
```

These jobs compare fused ERM, 3D-CNN-only ERM, GCN-GRU-only ERM, arousal MLDG,
and weak-adversary valence MLDG. They disable the decoder and VC auxiliary
regularizers so encoder/classifier learning can be established before adding
the subject-invariance and reconstruction objectives back.

For a conventional one-off four-subject smoke test per label:

```bash
TARGET_DIMENSION=valence sbatch src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/smoke_test_sic_mldg_brier_ablations.sh
TARGET_DIMENSION=arousal sbatch src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/smoke_test_sic_mldg_brier_ablations.sh
```

After both complete without NaNs, shape failures, or class collapse in the
first diagnostic rows, submit the full comparison. Its array runs the fused
model, GCN-GRU-only, and 3D-CNN-only profiles sequentially:

```bash
TARGET_DIMENSION=valence sbatch src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/full_run_sic_mldg_brier_ablations.sh
TARGET_DIMENSION=arousal sbatch src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/full_run_sic_mldg_brier_ablations.sh
```

API-v14/BiLSTM checkpoints are intentionally incompatible with this update;
retrain the LOSO population models before calibration.

The smoke and full launchers accept environment overrides for
`SMOKE_PROFILE`, `SOURCE_BATCH_SIZE`, `MLDG_TRIALS_PER_SUBJECT`, `USE_DECODER`,
`USE_SUBJECT_ADVERSARIAL`, `SUBJECT_ADVERSARIAL_WEIGHT`, `SUBJECT_LOSS_WEIGHT`,
`LEARNING_RATE`, `FOCAL_GAMMA`, `VC_BETA`, `VC_LAMBDA`, and
`CALIBRATION_USE_VC_TARGET`. Set `VC_BETA=grid` to retain the historical
two-value `[0.2, 0.6]` grid.
