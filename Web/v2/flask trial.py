from flask import Flask, render_template, request, redirect, json, jsonify

app = Flask(__name__)


@app.route('/')
def root():
    return "Test server"

@app.route("/index")
def index():
    return render_template('index.html')

# @app.route("/predictor")
# def predictor():
#     return render_template('predictor.html')

# @app.route("/prediction", methods=["GET","POST"])
# def prediction():
#     req = request.form
#     print(req)
#     age = int(request.form.get("age"))
#     # bmi = request.form.get("bmi")
#     # symptom_1 = request.form.get("symptom_1")
#     result=str(calculation(age))
#
#     return render_template('index.html',age=result)

def calculation(age):
    ans=age+10
    return ans

if __name__ == '__main__':
  app.run()