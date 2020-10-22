from flask import Flask, render_template, request, redirect, json, jsonify
import json
from joblib import load
import pandas as pd

app = Flask(__name__)

@app.route('/')
def root():
    return "Api test server"

@app.route("/prediction", methods=["GET","POST"])
def prediction():
    req = request.form
    print(req)
    #age = int(request.form.get("age"))
    #print(age)
    for key in req.keys():
        data = key
    #print(data)
    data_dic = json.loads(data)
    #print(data_dic.keys())

    bmi = data_dic['bmi'] #how to retreive value
    waist_to_hip_ratio = data_dic['waist_hip_ratio']
    bc_receptor = data_dic['bc_receptor']
    types_of_surgery = data_dic['surgery_type']
    num_lymph_nodes_removed = data_dic['lymph_removed']
    dash_score = data_dic['dash_score']
    ss2 = data_dic['ss2_hardness']

    if bmi and waist_to_hip_ratio and bc_receptor and types_of_surgery and num_lymph_nodes_removed and dash_score and ss2:
        #print('val1',bmi)
        #result=str(dummy_calculation(bmi))
        result=100*calculation(bmi, waist_to_hip_ratio, bc_receptor, types_of_surgery, num_lymph_nodes_removed, ss2, dash_score)
        result =str(round(result,1))
        resp_dic = {'result': result, 'msg': 'result performed'}
        resp = jsonify(resp_dic)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    resp = jsonify({'error': 'Missing data!'})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# model goes here, change function name as desired
def calculation(bmi, waist_to_hip_ratio, bc_receptor, types_of_surgery, num_lymph_nodes_removed, ss2, dash_score):
    classifier = load("classifier.mdl")
    return classifier.predict_proba([[bmi, waist_to_hip_ratio, bc_receptor, types_of_surgery, num_lymph_nodes_removed, ss2, dash_score]])[0,0]

def dummy_calculation(age):
    return age+10

if __name__ == '__main__':
    app.run(port=5001)
