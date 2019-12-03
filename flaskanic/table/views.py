from flask import Flask, render_template, request, session, \
    flash, redirect, url_for, abort
from jinja2 import TemplateNotFound
import json
from sklearn.externals import joblib as jl
import os

import sklearn
from flaskanic.views import read_feather_load_model, predictions_df
from . import table_page

###

df_test, enc_df_test, estimator = read_feather_load_model()
df_preds = predictions_df(df_test, enc_df_test, estimator)

#######

@table_page.route('/table', methods=['POST', 'GET'])
def full_predict():
    classes = ['table']
    df_table = df_preds.to_html(
        header="true", border=0, index=False, justify='left', 
        classes=classes, table_id='pred_table')
    return render_template("table.html", df_preds=df_table)