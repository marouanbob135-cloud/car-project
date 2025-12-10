from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model once
MODEL = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def get_embedding(img_pil):
    img = img_pil.resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    emb = MODEL.predict(x)[0]
    return emb
