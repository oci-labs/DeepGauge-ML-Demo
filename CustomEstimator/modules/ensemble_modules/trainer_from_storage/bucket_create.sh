BUCKET_NAME="custom_estimator"
echo $BUCKET_NAME
REGION=us-central1
DATA_PATH=/home/khodayarim/PycharmProjects/data/ImageEveryUnit

gsutil mb -l $REGION gs://$BUCKET_NAME/
