import os
import uuid

from flask import Flask, render_template, request, send_from_directory
from single_image import *
import numpy as np

app = Flask(__name__)

app.upload_path = "uploads"


def _preprocess(bbox, h, w):
    bboxes_np = np.array(
        [bbox['xmin'] * w, bbox['ymin'] * h, bbox['xmax'] * w, bbox['ymax'] * h, bbox['score']]).astype(int)

    if len(bboxes_np) == 0:
        bboxes_np = np.empty((0, 5))
    return bboxes_np


def draw_bbox(x, img, color=(0, 0, 255), label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory('uploads', filename)


@app.route('/static/images/<filename>')
def upload_static(filename):
    return send_from_directory('static/images/', filename)


@app.route("/")
def index():
    interface_files = []

    uploaded_files = [os.path.join("static", "images", i) for i in os.listdir(os.path.join("static", "images"))]
    for i, uploaded_file in enumerate(uploaded_files):

        img = cv2.imread(uploaded_file)

        output = single_image_predict(uploaded_file)

        h, w, _ = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Camera: {}".format(i), (40, 80), font, 3, (0, 255, 0), 3)
        for item in output:
            draw_bbox(x=_preprocess(item, h, w)[:4],
                      img=img,
                      color=(0, 0, 255))

        cv2.imwrite(uploaded_file, img)

        interface_files.append([uploaded_file, output])

    return render_template("index2.html", interface_files=interface_files)


@app.route("/", methods=["POST"])
def upload_files():
    if not os.path.exists(app.upload_path):
        os.makedirs(app.upload_path)

    interface_files = []

    uploaded_files = request.files.getlist("file")
    for i, uploaded_file in enumerate(uploaded_files):
        file_extension = os.path.splitext(uploaded_file.filename.lower())[-1]
        filename = str(uuid.uuid4()) + file_extension
        im_in_path = os.path.join(app.upload_path, filename)
        uploaded_file.save(im_in_path)

        output = single_image_predict(im_in_path)

        img = cv2.imread(im_in_path)
        h, w, _ = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Camera: {}".format(i), (40, 80), font, 3, (0, 255, 0), 3)
        for item in output:
            draw_bbox(x=_preprocess(item, h, w)[:4],
                      img=img,
                      color=(0, 0, 255))

        cv2.imwrite(im_in_path, img)

        interface_files.append([im_in_path, output])

    return render_template("index.html", interface_files=interface_files)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)