```
$ gcloud projects list
```

```
gcloud config set project MY_PROJECT
```
## Running Locally
Then set environment variables before starting your application:

```
export GOOGLE_CLOUD_PROJECT=[your-project-id]
export PUBSUB_VERIFICATION_TOKEN=[your-verification-token]
export PUBSUB_TOPIC=[your-topic]
python main.py
```

Application specific

```
export GOOGLE_CLOUD_PROJECT=ocideepgauge
export PUBSUB_VERIFICATION_TOKEN=7fSDp7HI29
export PUBSUB_TOPIC=flowers-prediction
python3 main.py
```

### Simulating push notifications
The application can send messages locally, but it is not able to receive push messages locally. You can, however, simulate a push message by making an HTTP request to the local push notification endpoint. There is an included sample_message.json. You can use curl to POST this:
```
$ curl -i --data @sample_message.json "http://localhost:8080/pubsub/push?token=7fSDp7HI29"
```

Deploy AppEngine into Project
```
gcloud app deploy --version pre-prod-1 --project MY_PROJECT
gcloud app deploy --version pre-prod-1 --project ocideepgauge
```
Browse the AppEngine Project
```
$ gcloud app browse -s deep-gauge
```
View any logs
```
 $ gcloud app logs tail
```
Create an isolated Python environment in a directory external to your project and activate it:
```
virtualenv env
source env/bin/activate
```
Navigate to your project directory and install dependencies:
```
cd YOUR_PROJECT
pip install  -r requirements.txt
```
Run the application:
```
python main.py
```
In your web browser, enter the following address:
```
http://localhost:8080
```
# Swagger, SQLLite, SQLAlchemy
## Install packages for database
```
pip3 install flask
pip3 install connexion
pip3 install flask_marshmallow
pip3 install connexion[swagger-ui]
pip3 install google-cloud-pubsub
pip3 install google-cloud-storage
pip3 install google-resumable-media
pip3 install marshmallow-sqlalchemy

```
# Models
Define a schema that represents that data.

```
User
  id
  user_name
  display_name
  company
  thumbnail
  updated

Device
  id  
  id_user
  name
  image
  bucket
  type
  location
  frame_rate
  refresh_rate
  notes
  high_threshold
  low_threshold
  updated

Setting
  id
  id_user
  type
  frame_rate
  refresh_rate
  updated

Reading
  id
  id_device
  prediction
  accuracy
  body
  timestamp
```
