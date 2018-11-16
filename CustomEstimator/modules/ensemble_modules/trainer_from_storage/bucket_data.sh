BUCKET_NAME="custom_estimator"
echo $BUCKET_NAME
REGION=us-central1
DATA_PATH=/home/khodayarim/PycharmProjects/data/ImageEveryUnit

## to copy the data folder to your Cloud Storage bucket.
gsutil cp -r $DATA_PATH gs://$BUCKET_NAME/data