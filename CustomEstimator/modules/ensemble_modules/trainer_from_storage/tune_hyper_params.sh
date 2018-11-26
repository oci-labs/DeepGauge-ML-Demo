##
REGION=us-central1
JOB_NAME=custom_estimator_tune_hyperParams_4
BUCKET="gs://custom_estimator"
##
PACKAGE_PATH="/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/trainer"
CONFIG_PATH="/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/hyperparam.yaml"
##
PRIMARY_PATH="${BUCKET}/misc/primary_models"
ENSEMBLE_PATH="${BUCKET}/misc/ensemble_graph"
IMG_PATH="${BUCKET}/data"
BIN_PATH="${BUCKET}/misc/logs/dumps"
EXPORT_PATH="${BUCKET}/misc/exported_model"
JOB_DIR="${BUCKET}/misc/logs/job_dir"

## --job-dir "$JOB_DIR" \
## --python-version 3.5 \

gcloud ml-engine jobs submit training "$JOB_NAME" \
    --stream-logs \
    --runtime-version 1.10 \
    --module-name trainer.task \
    --package-path "$PACKAGE_PATH" \
    --staging-bucket "$BUCKET" \
    --region "$REGION" \
    --config "$CONFIG_PATH" \
    -- \
    --verbosity="INFO" \
    --primary_models_directory="${PRIMARY_PATH}" \
    --ensemble_architecture_path="${ENSEMBLE_PATH}" \
    --path_to_images="${IMG_PATH}" \
    --export_dir="${EXPORT_PATH}" \
    --train_epochs=30 \
    --retrain_primary_models=False \
    --job_dir="$JOB_DIR"