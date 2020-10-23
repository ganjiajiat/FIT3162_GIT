from datetime import datetime
from flask import render_template, request, jsonify
from joblib import load
from src import app

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

# HTTP request on cmd: curl "https://<server:port>/predict_le_prob" -X POST -H "Content-Type: application/json" -d "{\"bmi\":25.18,\"waist_to_hip_ratio\":0.87,\"bc_receptor\":2,\"types_of_surgery\":1,\"num_lymph_nodes_removed\":1,\"ss2\":0,\"dash_score\":12.5}"
@app.route('/predict_le_prob', methods=['POST'])
def predict_le_prob():
    feature_values = request.get_json()
    bmi = feature_values['bmi']
    waist_to_hip_ratio = feature_values['waist_to_hip_ratio']
    bc_receptor = feature_values['bc_receptor']
    types_of_surgery = feature_values['types_of_surgery']
    num_lymph_nodes_removed = feature_values['num_lymph_nodes_removed']
    ss2 = feature_values['ss2']
    dash_score = feature_values['dash_score']
    classifier = load("D:/Dev/OnlineRepositories/FIT3162/FIT3162_GIT/Web/v2/prediction_api/project/FIT3162PredictionAPI/PredictionAPI/PredictionAPI/classifiers/classifier.mdl")
    prob = classifier.predict_proba([[bmi, waist_to_hip_ratio, bc_receptor, types_of_surgery, num_lymph_nodes_removed, ss2, dash_score]])[0,0]
    return jsonify(prob)