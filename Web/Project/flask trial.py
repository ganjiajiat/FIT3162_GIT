from flask import Flask, render_template,request, redirect

app = Flask(__name__)

@app.route('/')
def root():
    return "Test server"

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/predictor")
def predictor():
    return render_template('predictor.html')

@app.route("/prediction", methods=["GET","POST"])
def prediction():
    req = request.form
    print(req)
    age = request.form.get("age")
    bmi = request.form.get("bmi")
    symptom_1 = request.form.get("symptom_1")
    print(age)
    return render_template('index.html',age=age)

if __name__ == '__main__':
  app.run()