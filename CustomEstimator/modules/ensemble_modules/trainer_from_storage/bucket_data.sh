BUCKET_NAME="tt_ttt"
echo $BUCKET_NAME
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME/

## to copy the data folder to your Cloud Storage bucket.
gsutil cp -r data gs://$BUCKET_NAME/data