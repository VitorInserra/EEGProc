#!/bin/bash

set -euo pipefail

SCRIPT="src/eegproc/deep_learning/joint_architectures/SICModelv11/SLURM_scripts/smoke_test_sic_mldg_brier_ablations.sh"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: run this script from the EEGProc repository root."
    exit 2
fi

COMMON="CALIBRATION_EPOCHS=20,RECONSTRUCTION_LOSS_WEIGHT=0,USE_DECODER=false,VC_BETA=0,VC_LAMBDA=0,FOCAL_GAMMA=0,CALIBRATION_USE_VC_TARGET=false"

submit() {
    local job_name="$1"
    local settings="$2"
    sbatch \
        --job-name="$job_name" \
        --export="ALL,${COMMON},${settings}" \
        "$SCRIPT"
}

# 1. Establish whether the fused architecture learns emotion under ordinary
# empirical-risk minimization, without auxiliary objectives.
submit "sic_val_erm_fused" \
    "TARGET_DIMENSION=valence,TRAINING_METHOD=erm,SMOKE_PROFILE=full,SOURCE_EPOCHS=30,SOURCE_BATCH_SIZE=8,USE_SUBJECT_ADVERSARIAL=false"

# 2-3. Isolate each encoder branch under the same classification-only setup.
submit "sic_val_erm_cnn3d" \
    "TARGET_DIMENSION=valence,TRAINING_METHOD=erm,SMOKE_PROFILE=cnn3d_only,SOURCE_EPOCHS=30,SOURCE_BATCH_SIZE=8,USE_SUBJECT_ADVERSARIAL=false"
submit "sic_val_erm_gcn" \
    "TARGET_DIMENSION=valence,TRAINING_METHOD=erm,SMOKE_PROFILE=gcn_gru_only,SOURCE_EPOCHS=30,SOURCE_BATCH_SIZE=8,USE_SUBJECT_ADVERSARIAL=false"

# 4. Verify arousal with an episode size supported by its smallest
# subject/class trial pool.
submit "sic_aro_mldg_base" \
    "TARGET_DIMENSION=arousal,TRAINING_METHOD=mldg,SMOKE_PROFILE=full,SOURCE_EPOCHS=3,MLDG_STEPS_PER_EPOCH=100,MLDG_TRIALS_PER_SUBJECT=2,USE_SUBJECT_ADVERSARIAL=false"

# 5. Measure whether weak subject invariance helps after removing the stronger
# 0.6 adversarial encoder gradient used by the earlier collapsed runs.
submit "sic_val_mldg_adv01" \
    "TARGET_DIMENSION=valence,TRAINING_METHOD=mldg,SMOKE_PROFILE=full,SOURCE_EPOCHS=3,MLDG_STEPS_PER_EPOCH=100,MLDG_TRIALS_PER_SUBJECT=2,USE_SUBJECT_ADVERSARIAL=true,SUBJECT_ADVERSARIAL_WEIGHT=0.1,SUBJECT_LOSS_WEIGHT=1.0"
