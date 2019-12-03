from flask import Flask, render_template, request, session, \
    flash, redirect, url_for, abort, jsonify
from jinja2 import TemplateNotFound
from sklearn.externals import joblib as jl
import json
import pandas as pd
import os

# from keras.models import load_model
import sklearn
# import numpy as np
from random import sample
import flaskanic.views as fv
from . import predict_page

"""
For the json part:
https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/
"""


@predict_page.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        test_json_dump = json.dumps(request.get_json())
        df_test = pd.read_json(test_json_dump, orient='records')
    except Exception as e:
        raise e

    if df_test.empty:
        return(bad_request())
    else:
        print(df_test.head())
        enc_df_test = fv.preprocess_model(df_test)
        print(enc_df_test.head())

        _, _, estimator = fv.read_feather_load_model()
        predictions = estimator.predict(enc_df_test)

        df_preds = fv.predictions_df(df_test, enc_df_test, estimator)
        responses = jsonify(df_preds.to_json(orient='records'))
        responses.status_code = 200
    return (responses)


@predict_page.errorhandler(400)
def bad_request(error=None):
    message = {
            'status': 400,
            'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp