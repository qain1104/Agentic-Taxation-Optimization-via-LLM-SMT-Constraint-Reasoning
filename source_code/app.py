# app.py
from flask import Flask
from tax_calculators.tax_calculator import tax_bp

app = Flask(__name__)
app.register_blueprint(tax_bp, url_prefix='/api/tax')

if __name__ == '__main__':
    app.run(debug=True)
