from flask import Flask, render_template
from flaskanic import home_page

###

@home_page.route('/')
def home():
    return render_template("index.html")