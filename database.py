import json, os

DB_KNOWN = 'data/cars.json'
DB_UNKNOWN = 'data/unknown.json'

os.makedirs('data', exist_ok=True)
os.makedirs('static/known', exist_ok=True)
os.makedirs('static/unknown', exist_ok=True)

# initialize JSON files if not exist
if not os.path.exists(DB_KNOWN): json.dump([], open(DB_KNOWN,'w'))
if not os.path.exists(DB_UNKNOWN): json.dump([], open(DB_UNKNOWN,'w'))

# ------------------------
def load_known():
    return json.load(open(DB_KNOWN))

def load_unknown():
    return json.load(open(DB_UNKNOWN))

def save_known(db):
    json.dump(db, open(DB_KNOWN,'w'), indent=4)

def save_unknown(db):
    json.dump(db, open(DB_UNKNOWN,'w'), indent=4)

# ------------------------
def add_known_car(name, image_path, embedding, info=None):
    db = load_known()
    db.append({
        "name": name,
        "image": image_path,
        "embedding": embedding.tolist(),
        "info": info or {}
    })
    save_known(db)

def add_unknown_car(guess, image_path, embedding):
    db = load_unknown()
    db.append({
        "guess": guess,
        "image": image_path,
        "embedding": embedding.tolist()
    })
    save_unknown(db)
