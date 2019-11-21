# flaskanic

A simple **web interface** to a machine learning model of Titanic survivors based on the [Flask](https://flask.palletsprojects.com/) micro-framework.

Use your Titanic machine learning model to make predictions and display them as a html **data table**. You can tweak the code and use your own data and models to customize the app.

Serve the model as an API and get predictions by sending a **POST request** with JSON data from command line,
e.g.:

`curl -X POST -H "Content-Type: application/json" -d "@flaskanic/Titanic_data_sample.json" http://127.0.0.1:5000/predict`

## DISCLAIMER

flaskanic **has not been tested following a [TDD](https://en.wikipedia.org/wiki/Test-driven_development) approach**, so it's not guaranteed to be stable and is **not aimed to nor ready for production**.