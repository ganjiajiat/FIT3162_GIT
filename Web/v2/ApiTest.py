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
    print(data)
    data_dic = json.loads(data)
    print(data_dic.keys())

    bmi = data_dic['bmi'] #how to retreive value
    print('val1',bmi)
    result=str(dummy_calculation(bmi)) #
    #return render_template('index.html',age=result)

    resp_dic = {'result': result, 'msg': 'result performed'}
    resp = jsonify(resp_dic)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# model goes here, change function name as desired
def calculation(bmi, waist_to_hip_ratio, bc_receptor, types_of_surgery, num_lymph_nodes_removed, ss2, dash_score):
    classifier = load("classifier.mdl")
    return classifier.predict(
        pd.DataFrame(
           data=[[bmi, waist_to_hip_ratio, bc_receptor, types_of_surgery, num_lymph_nodes_removed, ss2, dash_score]],
           columns = ['BMI', 'Waist to hip datio', 'BC receptor', 'Types of surgery', 'Number of lymph nodes removed', 'SS2: Hardness/ difficulty finding shirts that fits', 'DASH score'] 
        )
    )[0]

def dummy_calculation(age):
    return age+10

if __name__ == '__main__':
    app.run(port=5001)
