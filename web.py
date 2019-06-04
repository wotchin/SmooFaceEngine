import cv2
from tempfile import SpooledTemporaryFile
import numpy as np
from flask import Flask
from flask import request
from utils.feature import get_feature_function
from utils.measure import cosine_similarity

model_path = "./trained_models/tiny_XCEPTION.hdf5"
get_feature = get_feature_function(model=model_path)

app = Flask(__name__, static_folder="web_static")
# if we save file to disk, we must use the following configuration.
# upload_folder = './web_static/uploads/'
# app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


@app.route("/")
def hello():
    return "Hello, SmooFaceEngine!"


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
                <input type="submit" value="upload">
            </form>
        </body>
        </html>
    '''
    return html


def get_feature_from_client(request_filename):
    # If we want to save this file to disk, we can use the
    # following code. But if we should save the binary file from client to disk,
    # the program would run slowly.
    """
    import random
    def get_random_string(length):
        string = ""
        for i in range(0, length):
            code = random.randint(97, 122)
            string += chr(code)
        return string

    pic = request.files[request_filename]
    img_type = pic.filename.split('.')[1]
    filename = get_random_string(10) + "." + img_type
    filepath = os.path.join(app.root_path,
                            app.config['UPLOAD_FOLDER'],
                            filename)
    pic.save(filepath)
    vector = get_feature(filepath)
    os.unlink(filepath)

    return vector
    """

    # the following codes:
    # We will save the file from client to memory, then
    # the program run much faster than saving it to disk.
    file = request.files[request_filename]
    stream = file.stream
    # for old version flask:
    """
     if isinstance(stream, SpooledTemporaryFile):
         stream = stream.file
    """
    value = bytearray(stream.read())
    value = np.asarray(value, dtype='uint8')
    img = cv2.imdecode(value, 1)
    vector = get_feature(img)
    return vector


@app.route("/data", methods=["POST"])
def predict():
    vector1 = get_feature_from_client('pic1')
    vector2 = get_feature_from_client('pic2')
    similarity = cosine_similarity(vector1, vector2)
    return str(similarity)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
