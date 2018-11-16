BUCKET_NAME="custom_estimator"
echo $BUCKET_NAME
REGION=us-central1
MISC_PATH=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/misc


## to copy the misc folder to your Cloud Storage bucket.
gsutil cp -r $MISC_PATH gs://$BUCKET_NAME/misc