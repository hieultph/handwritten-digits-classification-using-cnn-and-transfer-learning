from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('./trained/hwd.h5')

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imgfile = request.files['imagefile']
    image_path = './images/' + imgfile.filename
    imgfile.save(image_path)

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(28, 28))
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = np.array([image_arr])

    prediction = np.argmax(model.predict(image_arr))

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

