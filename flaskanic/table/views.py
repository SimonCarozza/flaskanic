from flask import Flask, render_template
from jinja2 import TemplateNotFound
import json
from sklearn.externals import joblib as jl
import pandas as pd
from pathlib import Path
import os

# from keras.models import load_model
import sklearn
import numpy as np
from random import sample
# autolrn classification's module
from autolrn.encoding import labelenc as lc
from flaskanic.table import table_page

###

df_test, enc_df_test, estimator = None, None, None

if not Path("./flaskanic/tmp/titanic_df").is_file():
    raise IOError("File not found")
else:
    df_test = pd.read_feather("./flaskanic/tmp/titanic_df")

if not Path("./flaskanic/tmp/enc_titanic_df").is_file():
    raise IOError("File not found")
else:
    enc_df_test = pd.read_feather("./flaskanic/tmp/enc_titanic_df")

ml_model_path = "./flaskanic/models/titan_LogRClf_2nd_light_opt_0525.pkl"

if not Path(ml_model_path).is_file():
    raise IOError("File not found")
else:
    estimator = jl.load(ml_model_path)

print(estimator)
# predictions = estimator.predict(enc_df_test)


###

def predictions_df(df=None, enc_df=None, estimator=None):
    # feat selection to get most important features (only for n_columns > 10)
    mask = estimator.named_steps['feature_selection'].get_support(indices=True)
    print(mask)
    if df is not None:
        df_preds = df.iloc[:, mask]
    # print(df_preds.head(2))
    print(df_preds.shape[1])
    if enc_df is not None and estimator is not None:
        predictions = estimator.predict(enc_df)
    df_preds["Survived"] = pd.Series(predictions)
    if hasattr(estimator, "predict_proba"):
        pred_probas=estimator.predict_proba(enc_df)[:,1]*100
        df_preds["(%)"]=pd.Series(np.around(pred_probas, 1))
    return df_preds

df_preds = predictions_df(df_test, enc_df_test, estimator)

#######

@table_page.route('/table', methods=['POST', 'GET'])
def full_predict():
    # df_preds = predictions_df(df_test, enc_df_test, estimator)
    classes = ['table']
    df_table = df_preds.to_html(
        header="true", border=0, index=False, justify='left', 
        classes=classes, table_id='pred_table')
    return render_template("table.html", df_preds=df_table)