```
$ gcloud projects list
```

```
gcloud config set project MY_PROJECT
```

Deploy AppEngine into Project
```
gcloud app deploy --version pre-prod-1 --project MY_PROJECT
```
Browse the AppEngine Project
```
$ gcloud app browse -s pet-products
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
