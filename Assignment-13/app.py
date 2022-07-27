from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras
from playsound import playsound

labels = {
    0: 'Too Early',
    1: 'Rippen',
    2: 'Too Late'
}

model = keras.models.load_model('model.h5')


img = Image.open('test.jpg')


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open('test.jpg')  # test.jpg
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
image_array = np.asarray(image)
normalise_image_array = (image_array.astype(np.float32)/127.0)-1
data[0] = normalise_image_array
prediction = model.predict(data)
print(prediction)  # [[0.5,0.5,0.7,0.3]]
# Decision Logic
prediction = list(prediction[0])
max_prediction = max(prediction)
index_max = prediction.index(max_prediction)
print(index_max)
result = labels[index_max]
if result == "Rippen":
    playsound("buzzer.wav")