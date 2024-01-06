from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
import numpy as np
import cv2
import os


app = Flask(__name__)
model = load_model('model.hdf5')

app.config['UPLOAD_FOLDER'] = 'images'


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = os.path.join("images", imagefile.filename) #"./images/" + imagefile.filename
    imagefile.save(image_path)

    file_name = imagefile.filename
    # image = load_img(image_path, target_size=(224, 224))
    # image = img_to_array(image)
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))



    image = cv2.imread(image_path)
    image = cv2.resize(image, (224,224), interpolation = cv2.INTER_NEAREST)
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = np.array(image)
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image.reshape((1,224,224,3))

    # image = preprocess_input(image)
    yhat = model.predict(image)
    print(yhat)
    print(type(yhat))
    classes=['Normal','Cataract', 'Glaucoma', 'AMD','Abnormal']

    # classification = classes[np.argmax(yhat)]
    # label = decode_predictions(yhat)
    # label = label[0][0]
    normal = int(yhat[0,0]*100)
    cataract = int(yhat[0,1]*100)
    glaucoma = int(yhat[0,2]*100)
    amd = int(yhat[0,3]*100)
    abnormal = int(yhat[0,4]*100)

    classification = classes[np.argmax(yhat)] #'%s (%.2f%%)' % (label[1], label[2]*100)

    # prob_classe = yhat[0]
    # print(prob_classe)

    

    # diseas_pred = dict(zip(classes,prob_classe))
    # print(diseas_pred)
    # has = str(diseas_pred)


    return render_template('index.html',fileN = file_name, prediction=classification,norma = normal,cata=cataract,gulu= glaucoma,am = amd, abnor = abnormal,image_path=image_path)


if __name__ == '__main__':
    app.run(port=3000, debug=True)