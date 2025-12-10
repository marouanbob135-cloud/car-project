from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import os
import numpy as np
from numpy.linalg import norm
from database import load_known, add_known_car, add_unknown_car
from embeddings import get_embedding

app = Flask(__name__)

# -----------------------------
def cosine(a, b):
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))

# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory('static', filename)

# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    emb = get_embedding(img)

    known = load_known()
    best_score = 0
    best_car = None

    for car in known:
        score = cosine(emb, np.array(car['embedding']))
        if score > best_score:
            best_score = score
            best_car = car

    # threshold for known
    if best_score > 0.7:
        return jsonify({"status":"found","car":best_car})

    # unknown: save
    idx = len(load_known()) + len(load_unknown())
    path = f"static/unknown/unk_{idx}.jpg"
    img.save(path)
    add_unknown_car("Unknown Car", path, emb)

    return jsonify({"status":"unknown","guess":"Unknown Car","image":path})

# -----------------------------
@app.route("/admin")
def admin_dashboard():
    unknowns = load_unknown()
    return render_template("admin.html", unknowns=unknowns)

# -----------------------------
@app.route("/admin/add", methods=["POST"])
def admin_add():
    name = request.form['name']
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    emb = get_embedding(img)
    path = f"static/known/{name.replace(' ','_')}.jpg"
    img.save(path)
    add_known_car(name, path, emb)
    return jsonify({"status":"added","name":name})

# -----------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

