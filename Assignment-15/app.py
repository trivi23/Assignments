import streamlit as st
import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])


def load_image(img):
    return Image.open(img)


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(150, 150))
    img = np.reshape(img, [-1, 150, 150, 1])

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(img)
    preds = preds.reshape(1, -1)[0]
    print(preds)

    if preds[0] == 0:
        preds = "Pneumonia Detected"
    elif preds[0] == 1:
        preds = "Normal"

    return preds


st.title('Pneumonia Detection using Deep Learning')

src_file = st.file_uploader('Upload Images', type=['png', 'jpg', 'jpeg'])

if src_file is not None:
    file_details = {}
    file_details['size'] = src_file.size
    file_details['type'] = src_file.type
    file_details['name'] = src_file.name
    st.write(file_details)
    st.image(load_image(src_file), width=250)

    with open(os.path.join('uploads', 'src.jpeg'), 'wb') as f:
        f.write(src_file.getbuffer())

    st.success('file saved')
    st.warning('Processing Image')

    preds = model_predict('uploads/src.jpeg', model)
    result = preds
    st.success('Pneumonia Output: '+result)
