import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mp

model = tf.keras.models.load_model("saved_model/mdl.hdf5")
confidence_threshold = 0.65

with open('style.css')as f:
 st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

st.markdown("<h1 style='text-align: center;'>Image Classification</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        color: Aliceblue;
        background-image: linear-gradient(to bottom left, #3B2667,#BC78EC) !important;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    
    [data-testid="stVerticalBlock"]{
        background-color: linear-gradient(to bottom left, #2193b0 ,#6dd5ed) !important;
        
        padding-bottom: 40px;
    }
    .predict-button {
        background-color: #FF5722;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        font-size: 18px;
        cursor: pointer;
    }
    [data-testid="stFileUploader"]{
        background-color: linear-gradient(to bottom left, #2193b0 ,#6dd5ed) !important;
        
        padding-bottom: 40px;
     }
     [data-testid="stButton"]{
        background-color: linear-gradient(to bottom left, #2193b0 ,#6dd5ed) !important;
        
        padding-bottom: 40px;
     }  
    
    </style>
    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Choose an Image", type=["jpg","png","jpeg"])

map_dict = {0: 'dog',
            1: 'horse',
            2: 'elephant',
            3: 'butterfly',
            4: 'chicken',
            5: 'cat',
            6: 'cow',
            7: 'sheep',
            8: 'spider',
            9: 'squirrel'}


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image,(224,224))
    st.image(image, channels="RGB")

    resized = mp(resized)
    img_reshape = resized[np.newaxis,...]

    pred = st.button("Predict", key="predict_button", help="Click to predict the animal")

    if pred:
        prediction = model.predict(img_reshape)
        pred_class = prediction.argmax()
        confidence = prediction[0][pred_class]

        if confidence > confidence_threshold:
            st.success("The animal in the given image is {}".format(map_dict[pred_class]))
        else:
            st.warning("No recognizable animal found in the image.")