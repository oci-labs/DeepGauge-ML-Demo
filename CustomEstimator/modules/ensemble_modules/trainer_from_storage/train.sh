gcloud ml-engine jobs submit training "$JOB_ID" \
  --stream-logs \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --runtime-version=1.4 \
  -- \
  --output_path "${GCS_PATH}/training" \
  --eval_data_paths "${GCS_PATH}/preproc/eval*" \
  --train_data_paths "${GCS_PATH}/preproc/train*"


##
REGION=us-central1
JOB_NAME=deep_gauge_1
BUCKET=gs://deep_gauge
PACKAGE_PATH=CustomEstimator/modules/ensemble_modules/trainer_from_storage/trainer


gcloud ml-engine jobs submit training $JOB_NAME \
    --runtime-version 1.8 \
    --module-name trainer.task \
    --package-path "$PACKAGE_PATH" \
    --staging-bucket "$BUCKET" \
    --region $REGION \
    -- \
    --verbosity DEBUG