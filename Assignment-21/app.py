from tensorflow import keras

model = keras.models.load_model('h2')

a = model.predict([[4,4]])

print(a)

