##
REGION=us-central1
JOB_NAME=custom_trial_131
BUCKET="gs://custom_estimator"
##
PACKAGE_PATH="/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/trainer"
##
PRIMARY_PATH="${BUCKET}/misc/primary_models"
ENSEMBLE_PATH="${BUCKET}/misc/ensemble_graph"
IMG_PATH="${BUCKET}/data"
BIN_PATH="${BUCKET}/misc/logs/dumps"
STAGING_BUCKET="gs://custom_estimator_staging_bucket"
JOB_DIR="${BUCKET}/misc/logs/job_dir/${JOB_NAME}"

## --scale-tier BASIC_GPU

gcloud ml-engine jobs submit training "$JOB_NAME" \
    --scale-tier BASIC_GPU \
    --stream-logs \
    --runtime-version 1.10 \
    --module-name trainer.task \
    --package-path "$PACKAGE_PATH" \
    --staging-bucket "$STAGING_BUCKET" \
    --region "$REGION" \
    -- \
    --verbosity="INFO" \
    --primary_models_directory="${PRIMARY_PATH}" \
    --ensemble_architecture_path="${ENSEMBLE_PATH}" \
    --path_to_images="${IMG_PATH}" \
    --dev=False \
    --train_epochs=500 \
    --batch_size=400 \
    --retrain_primary_models=True \
    --learning_rate=8e-4 \
    --job_dir="$JOB_DIR" \
