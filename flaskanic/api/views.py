from flask import Flask, render_template, request, jsonify
from jinja2 import TemplateNotFound
from sklearn.externals import joblib as jl
import json
import pandas as pd
import os

# from keras.models import load_model
import sklearn
# import numpy as np
from random import sample
# autolrn classification's module
from autolrn.encoding import labelenc as lc
from flaskanic.table.views import predictions_df
from flaskanic.api import predict_page

"""
For the json part:
https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/
"""


@predict_page.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        test_json_dump = json.dumps(request.get_json())
        test_df = pd.read_json(test_json_dump, orient='records')
    except Exception as e:
        raise e

    if test_df.empty:
        return(bad_request())
    else:
        print(test_df.head())
        if 'Cabin' in test_df:
            test_df.drop(['Cabin'], axis=1, inplace=True)

        # replace missing valus in 'Age', 'Embarked'
        if test_df.isnull().values.any():
            print("Null values here... replacing them.")
            df_test.fillna(method='pad', inplace=True)
            df_test.fillna(method='bfill', inplace=True)

        enc_df_test = lc.dummy_encode(test_df)
        print(enc_df_test.head())
        estimator = jl.load("./flaskanic/models/titan_LogRClf_2nd_light_opt_0525.pkl")
        # predictions = estimator.predict(enc_df_test)

        df_preds = predictions_df(test_df, enc_df_test, estimator)
        responses = jsonify(df_preds.to_json(orient='records'))
        responses.status_code = 200
    return (responses)


@predict_page.errorhandler(400)
def bad_request(error=None):
    message = {
            'status': 400,
            'message': 'Bad Request: ' + request.url 
            + ', Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp