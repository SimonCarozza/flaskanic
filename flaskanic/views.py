from flask import Flask, Blueprint, render_template
from flaskanic import home_page
from sklearn.externals import joblib as jl
from pathlib import Path
from pandas import read_feather
import numpy as np
# autolrn classification's module
from autolrn.encoding import labelenc as lc

###

def preprocess_model(df_test=None):
    # print("Dropping 'Cabin' column -- too many missing values")
    if df_test is not None:
        if 'Cabin' in df_test:
            df_test.drop(['Cabin'], axis=1, inplace=True)

        # replace missing valus in 'Age', 'Embarked' -you could just use median()
        if df_test.isnull().values.any():
            print("Null values here... replacing them.")
            df_test.fillna(method='pad', inplace=True)
            df_test.fillna(method='bfill', inplace=True)

        enc_df_test = lc.dummy_encode(df_test)
    else:
        raise TypeError("Please pass in a non-empty pandas Dataframe")

    return enc_df_test


def read_feather_load_model():
    if not Path("./flaskanic/tmp/titanic_df").is_file():
        raise FileNotFoundError("File not found")
    else:
        df_test = read_feather("./flaskanic/tmp/titanic_df")

    if not Path("./flaskanic/tmp/enc_titanic_df").is_file():
        raise FileNotFoundError("File not found")
    else:
        enc_df_test = read_feather("./flaskanic/tmp/enc_titanic_df")

    # load 0525 to compare table widths
    ml_model = "./flaskanic/models/titan_LogRClf_2nd_light_opt_0987.pkl"

    estimator = None

    try:
        jl.load(ml_model)
    except FileNotFoundError as fnfe:
        raise fnfe
    except Exception as e:
        raise e
    else:
        estimator = jl.load(ml_model)
        print(estimator)

    # estimator = jl.load(ml_model)

    return df_test, enc_df_test, estimator


def predictions_df(df=None, enc_df=None, estimator=None, threshold=0.5):
    # feat selection to get most important features (only for n_columns > 10)
    mask = estimator.named_steps['feature_selection'].get_support(indices=True)
    print(mask)
    if df is not None:
        df_preds = df.iloc[:, mask].copy()
    else:
        raise TypeError("Please pass in a non-empty pandas Dataframe")
    # print(df_preds.head(2))
    if enc_df is not None and estimator is not None:
        predictions = estimator.predict(enc_df)
    else:
        raise TypeError(
            "Please pass a non-empty pandas Dataframe and valid sklearn "
            "estimator")

    if hasattr(estimator, "predict_proba"):
        pred_probas=estimator.predict_proba(enc_df)[:,1]*100
        if threshold != 0.5:
            df_preds["Survived"] = np.where(pred_probas/100 > threshold, 1, 0)
        else:
            df_preds["Survived"] = predictions
        df_preds["(%)"] = np.around(pred_probas, 1)
    else:
        df_preds["Survived"] = predictions

    return df_preds

###

@home_page.route('/')
def home():
    return render_template("index.html")