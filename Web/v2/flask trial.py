from flask import Flask, render_template, request, redirect, json, jsonify

app = Flask(__name__)


@app.route('/')
def root():
    return "Test server"

@app.route("/index")
def index():
    return render_template('index.html')



if __name__ == '__main__':
  app.run()
