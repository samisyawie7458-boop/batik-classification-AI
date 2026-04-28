from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 🔥 pastikan folder upload ada
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 🔥 load model
model = load_model("batik_model_v1.h5")

classes = [
    "batik-bali","batik-betawi","batik-celup","batik-cendrawasih",
    "batik-ceplok","batik-ciamisl","batik-garutan","batik-gentongan",
    "batik-kawung","batik-keraton","batik-lasem","batik-megamendung",
    "batik-parang","batik-pekalongan","batik-priangan","batik-sekar",
    "batik-sidoluhur","batik-sidomukti","batik-sogan","batik-tambal"
]

def predict_image(img_path):
    img = cv2.imread(img_path)

    # ❗ cek kalau gambar gagal dibaca
    if img is None:
        return "Gambar tidak valid", 0

    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    pred = model.predict(img, verbose=0)
    class_index = np.argmax(pred)
    confidence = np.max(pred) * 100

    return classes[class_index], confidence


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
         
            filename = file.filename.replace(" ", "_")
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(path)

            result, confidence = predict_image(path)
            image_path = path

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)