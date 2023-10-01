import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

model = keras.models.load_model("skinCancerClassification.h5")

class_labels = {
    0: 'dermatofibroma',
    1: 'melanoma',
    2: 'nevus',
    3: 'seborrheic keratosis',
    4: 'squamous cell carcinoma',
    5: 'pigmented benign keratosis',
    6: 'basal cell carcinoma',
    7: 'vascular lesion',
    8: 'actinic keratosis'
}

def classify_skin_cancer(image):
    # Preprocess the image
    image = np.array(image)
    image = tf.image.resize(image, (75, 100))  
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    class_name = class_labels[class_index]

    confidence = np.max(predictions)

    return f"Predicted Class: {class_name}\nConfidence: {confidence:.2f}"

iface = gr.Interface(
    fn=classify_skin_cancer,
    inputs="image",
    outputs="text",
    live=True,
    title='<h1 style="text-align: center;">Skin Cancer Classification! 🌻</h1>',
    description=(
        "<h2><b>Explore Skin Cancer Image Classification!</b></h2>"
        "<p>Join me in the world of skin health and medical innovation. " 
        "Be part of a game-changing journey where you can support healthcare, " 
        "make a real difference, and impact lives. 🌍🩺🤝 " 
        "Discover the power of AI in skin cancer diagnosis. Start exploring now!</p>"
    )
)

iface.launch()
