import os

from flask import Flask
from flask import request
from utils.feature import get_feature_function
from utils.measure import cosine_similarity

upload_folder = './web_static/uploads/'
model_path = "./trained_models/tiny_XCEPTION.hdf5"
get_feature = get_feature_function(model=model_path)

app = Flask(__name__, static_folder="web_static")
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


@app.route("/")
def hello():
    return "Hello, SmooFaceNet"


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
    pic1_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], pic1.filename)
    pic2_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], pic2.filename)
    pic1.save(pic1_path)
    pic2.save(pic2_path)
    vector1 = get_feature(pic1_path)
    vector2 = get_feature(pic2_path)
    similarity = cosine_similarity(vector1, vector2)
    os.unlink(pic1_path)
    os.unlink(pic2_path)
    return str(similarity)


if __name__ == "__main__":
    if not os.path.exists(upload_folder):
        os.system("mkdir -p " + upload_folder)  # ONLY FOR *nix, NOT FOR Windows
    app.run(host='0.0.0.0', port=8080, debug=True)
