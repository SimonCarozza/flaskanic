from pandas import read_csv
from pathlib import Path
from autolrn.encoding import labelenc as lc
import os
from flask import Flask, Blueprint, render_template

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

	from flaskanic.views import preprocess_model, home_page
	enc_df_test = preprocess_model(df_test)

	os.makedirs('./flaskanic/tmp', exist_ok=True)
	if not Path("./flaskanic/tmp/titanic_df").is_file():
	    df_test.to_feather("./flaskanic/tmp/titanic_df")
	if not Path("./flaskanic/tmp/enc_titanic_df").is_file():
		enc_df_test.to_feather("./flaskanic/tmp/enc_titanic_df")

	from flaskanic.table.views import table_page
	from flaskanic.api.views import predict_page

	# register blueprints
	app.register_blueprint(home_page)
	app.register_blueprint(table_page)
	app.register_blueprint(predict_page)