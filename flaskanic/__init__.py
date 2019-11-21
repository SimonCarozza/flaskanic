from pandas import read_csv
from pathlib import Path
from autolrn.encoding import labelenc as lc
import os
from flask import Flask, render_template, Blueprint

app = Flask(__name__)

home_page = Blueprint('index', __name__,
	template_folder='templates',
	static_folder='static')

with app.app_context():

	# create and save pre-processed dataframe
	names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp',
         'Parch','Ticket','Fare','Cabin','Embarked']

	df_test = read_csv(
	    "./flaskanic/datasets/titanic_test.csv", delimiter=",",
	    # header=0, names=names,
	    na_values={'Age': '', 'Cabin': '', 'Embarked': ''},
	    dtype={'Name': 'category', 'Sex': 'category',
	           'Ticket': 'category', 'Cabin': 'category',
	           'Embarked': 'category'})

	# print("Dropping 'Cabin' column -- too many missing values")
	df_test.drop(['Cabin'], axis=1, inplace=True)

	# replace missing valus in 'Age', 'Embarked'
	if df_test.isnull().values.any():
	    print("Null values here... replacing them.")
	    df_test.fillna(method='pad', inplace=True)
	    df_test.fillna(method='bfill', inplace=True)

	enc_df_test = lc.dummy_encode(df_test)

	os.makedirs('./flaskanic/tmp', exist_ok=True)
	if not Path("./flaskanic/tmp/titanic_df"):
	    df_test.to_feather("./flaskanic/tmp/titanic_df")
	if not Path("./flaskanic/tmp/enc_titanic_df"):
		enc_df_test.to_feather("./flaskanic/tmp/enc_titanic_df")

	# import blueprints
	from flaskanic.views import home_page
	from flaskanic.table.views import table_page
	from flaskanic.api.views import predict_page

	# register blueprints
	app.register_blueprint(home_page)
	app.register_blueprint(table_page)
	app.register_blueprint(predict_page)