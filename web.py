import os

from flask import Flask
from flask import request
import numpy as np
import keras
from flask import url_for
from feature import feature

app = Flask(__name__, static_folder="web_static")
app.config['UPLOAD_FOLDER'] = './web_static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
model = feature(model_path="./trained_models/smooFace.28-0.996528.hdf5")


@app.route("/")
def hello():
    return "Hello ,SmooFaceNet"


@app.route("/test")
def test():
    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Uploading</title>
</head>
<body>
    <form action="/data" method="post" enctype="multipart/form-data">
        <input type="file" name="pic1" value="Pic1" /><br>
        <input type="file" name="pic2" value="Pic2" /><br>
        <input type="submit" value="上传">
    </form>
</body>
</html>
    '''
    return html


@app.route("/data", methods=["POST"])
def predict():
    pic1 = request.files['pic1']
    pic2 = request.files['pic2']
    pic1.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], pic1.filename))
    pic2.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], pic2.filename))
    vector1 = model.get_vector(pic1.filename)
    vector2 = model.get_vector(pic2.filename)
    similarity = model.cosine_similarity(vector1,vector2)
    os.unlink(pic1.filename)
    os.unlink(pic2.filename)
    return similarity


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
