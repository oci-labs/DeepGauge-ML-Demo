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
python main.py
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
