BUCKET_NAME="custom_estimator"
echo $BUCKET_NAME
REGION=us-central1
MISC_PATH=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/misc
DATA_PATH=/home/khodayarim/PycharmProjects/data/ImageEveryUnit
# SETUP_PATH=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/setup.py

gsutil mb -l $REGION gs://$BUCKET_NAME/

## to copy the misc and data folders to your Cloud Storage bucket.
gsutil cp -r $MISC_PATH gs://$BUCKET_NAME/misc
gsutil cp -r $DATA_PATH gs://$BUCKET_NAME/data
# gsutil cp -r $SETUP_PATH gs://$BUCKET_NAME/data