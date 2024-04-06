import cv2
from flask import Flask, render_template, Response,request,redirect

app = Flask(__name__)

cameras=dict()

@app.route('/')
def dash():
    return render_template('dash.html',cameras=cameras)

if __name__ == "__main__":
    app.run(debug=True)