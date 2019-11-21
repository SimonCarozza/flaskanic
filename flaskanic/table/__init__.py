from flask import Blueprint

table_page = Blueprint(
    'table', __name__,
    template_folder='templates',
    static_folder='static'
    )