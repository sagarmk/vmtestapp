#!flask/bin/python
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import load_model
from urllib.request import urlopen
import requests
from flask import Flask
from flask import request
import pandas as pd
from sklearn import linear_model
import pickle
import jsonify

app = Flask(__name__)


print(" Loading vgg16 keras DNN model > ")
from keras.applications.vgg16 import VGG16
model = VGG16()
#model = load_model('model.h5')
print(" Model loaded ")

@app.route('/')
def index():
    return " <h1> Flask is running </h1> "


@app.route('/api')
def api():
	return " <h1> API is running "



@app.route('/predict', methods=['GET'])


def predict():
    
    my_var = request.args.get('url', None)

    

    def predictor(url):
    
    
        
        #load image from url
        file = urlopen(url)
        
        # load an image from file
        image = load_img(file, target_size=(224, 224))


        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # predict the probability across all output classes
        yhat = model.predict(image)
        # convert the probabilities to class 
        
        label = decode_predictions(yhat)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0]
        # print the classification
        #return('%s (%.2f%%)' % (label[1], label[2]*100))
	    #alpha = [label[1],label[2]]
        pred={"predicted label":label[1],"confdence":label[2]}
	    #predictions = label
	jsonpred = jsonify(pred)
        return(jsonpred)
       	#return (label[2][1])

    predicted_value = predictor(my_var)

    return str(predicted_value)




if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0')    
