from flask import Flask, render_template, request, redirect, json, jsonify

app = Flask(__name__)


@app.route('/')
def root():
    return "Test server"

@app.route("/index")
def index():
    return render_template('indexv2.html')



if __name__ == '__main__':
  app.run()
