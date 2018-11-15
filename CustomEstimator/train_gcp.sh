##
REGION=us-central1
JOB_NAME=training_trial_1
BUCKET=gs://custom_estimator
##
JOB_DIR="${BUCKET}/misc/logs/job_dir"
PACKAGE_PATH=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/trainer
##
PRIMARY_PATH="${BUCKET}/misc/primary_models"
ENSEMBLE_PATH="${BUCKET}/misc/temporary_models"
IMG_PATH="${BUCKET}/data"
BIN_PATH="${BUCKET}/misc/logs/dumps"
EXPORT_PATH="${BUCKET}/misc/exported_model"


##
gcloud ml-engine jobs submit training "$JOB_NAME" \
    --stream-logs \
    --runtime-version 1.10 \
    --module-name trainer.task \
    --package-path "$PACKAGE_PATH" \
    --python-version 3.5 \
    --staging-bucket "$BUCKET" \
    --job-dir "$JOB_DIR"  \
    --region "$REGION" \
    -- \
    --verbosity="DEBUG" \
    --primary_models_directory="$PRIMARY_PATH" \
    --ensemble_architecture_path="$ENSEMBLE_PATH" \
    --path_to_images="$IMG_PATH" \
    --bin_path="$BIN_PATH" \
    --export_dir="$EXPORT_PATH"

