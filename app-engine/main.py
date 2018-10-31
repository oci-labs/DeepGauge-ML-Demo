# [START gae_python37_render_template]
import datetime
import pandas as pd
from flask import Flask, Response, request, json, render_template

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
# [START gae_python37_render_template]
