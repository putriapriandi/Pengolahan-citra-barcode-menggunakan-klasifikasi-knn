import os
import urllib.request
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
from flask.templating import render_template
import base64
from flask import Flask, Response, request
import json
import pandas as pd
import numpy as np
import os
import math
import operator
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, "uploads/")
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    base64_img = request.form.get("IMG")
    '''
    print("Start Print")
    print(base64_img)
    print("End of Print")
    '''

    base64_img_bytes = base64_img.encode('utf-8')
    filename = 'image_from_android.png'
    destination = "/".join([target, filename])
    with open(destination, 'wb') as file_to_save:
        decoded_image_data = base64.decodebytes(base64_img_bytes)
        file_to_save.write(decoded_image_data)
        img = cv2.imread('c:/Users/Lenovo/PycharmProjects/yo/uploads/image_from_android.png')

        # show image format (basically a 3-d array of pixel color info, in BGR format)

        # %%

        # convert image to RGB color for matplotlib
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # show image with matplotlib

        # convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # grayscale image represented as a 2-d array

        # have to convert grayscale back to RGB for plt.imshow()

        # find average per row
        # np.average() takes in an axis argument which finds the average across that axis.
        average_color_per_row = np.average(img, axis=0)

        # find average across average per row
        average_color = np.average(average_color_per_row, axis=0)

        # convert back to uint8
        average_color = np.uint8(average_color)

        # create 100 x 100 pixel array with average color value
        average_color_img = np.array([[average_color] * 100] * 100, np.uint8)

        # threshold for grayscale image
        _, threshold_img = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY)

        threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
        plt.figure(figsize=(10, 10))

        # %%

        # edge detection
        dst = cv2.Canny(threshold_img, 60, 200, None, 3)

        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 100, 100)

        import math

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, None, 50, 100)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 1, cv2.LINE_AA)

        width = 50
        height = 50
        dim = (width, height)
        # resize image
        res = cv2.resize(cdstP, dim, interpolation=cv2.INTER_AREA)
        res.resize(10, 50)
        res.flatten()
        _, threshold_img = cv2.threshold(res, 30, 1, cv2.THRESH_BINARY)
        print(threshold_img.flatten())

        df = pd.read_csv('go.csv')
        df.replace('?', -99999, inplace=True)
        df.drop(['id'], 1, inplace=True)

        X = np.array(df.drop(['label'], 1))
        y = np.array(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf = neighbors.KNeighborsClassifier(n_neighbors=1)

        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        # threshold_img=threshold_img.reshape(len(threshold_img),-1)
        # threshold_img= np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # example_measures = np.array([[0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
        # np.array2string(threshold_img, precision=2, separator=',', suppress_small=True)
        threshold_img = np.array([threshold_img.flatten()])
        # example_measures = example_measures.reshape(len(example_measures),-1) #-1
        threshold_img = threshold_img.reshape(len(threshold_img), -1)  # -1
        prediction = clf.predict(threshold_img)  # example

        if prediction == [1]:
            harga = 5800
            prediction = 'Ultra Milk (24x220 Ml)'
        if prediction == [2]:
            harga = 2800
            prediction = 'Buavita'
        if prediction == [3]:
            harga = 4800
            prediction = 'popmie'
        if prediction == [4]:
            harga = 300
            prediction = 'hansaplast'
        if prediction == [5]:
            harga = 6800
            prediction = 'Leminerale'
        if prediction == [6]:
            harga = 2800
            prediction = 'rexona'
        if prediction == [7]:
            harga = 2800
            prediction = 'Pocky'
        if prediction == [8]:
            harga = 2800
            prediction = 'cdr'
        if prediction == [9]:
            harga = 2800
            prediction = 'panadol'
        if prediction == [10]:
            harga = 2800
            prediction = 'rebo'
        if prediction == [11]:
            harga = 2800
            prediction = 'Walini'
        if prediction == [12]:
            harga = 2800
            prediction = 'Maizena'
        if prediction == [13]:
            harga = 2800
            prediction = 'Bye bye fever'
        if prediction == [14]:
            harga = 2800
            prediction = 'koyo'
        if prediction == [15]:
            harga = 2800
            prediction = 'dove'
        print(prediction)
    return prediction
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    json_dump_knn = json.dumps({'Nama Barang': prediction}, cls=NumpyEncoder)

    msg = {
        "Prediksi": json_dump_knn,
        "harga": harga
    }

    resp = Response(response=json.dumps(msg),
                    status=200,
                    mimetype="application/json")

    return json_dump_knn



# @app.route('/postjson', methods=['GET'])
# def postJsonHandler():
#     print(request.is_json)
#     content = request.get_json()
#     print(content)
#     return print("berhasil")



#
if (__name__) == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
