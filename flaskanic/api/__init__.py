from flask import Blueprint

predict_page = Blueprint(
    'api', __name__,
    template_folder='templates',
    static_folder='static'
    )
