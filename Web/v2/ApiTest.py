from flask import Flask, render_template, request, redirect, json, jsonify
import json

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
    result=str(calculation(bmi)) #
    #return render_template('index.html',age=result)

    resp_dic = {'result': result, 'msg': 'result performed'}
    resp = jsonify(resp_dic)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# model goes here, change function name as desired
def calculation(age):
    # dummy calculation, to replace with final model
    ans=age+10
    return ans

if __name__ == '__main__':
    app.run(port=5001)