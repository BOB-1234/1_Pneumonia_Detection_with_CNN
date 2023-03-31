import keras.utils as image
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Model

from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from glob import glob
import keras.utils as image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#loading model
model=load_model('patient_xrays.h5')

#loading the image from files can be changed for testing different files unhealthy images
img=image.load_img('Datasets/test/PNEUMONIA/person1946_bacteria_4874.jpeg',target_size=(224,224))
#img=image.load_img('Datasets/test/PNEUMONIA/person1946_bacteria_4875.jpeg',target_size=(224,224))
#img=image.load_img('Datasets/test/PNEUMONIA/person1947_bacteria_4876.jpeg',target_size=(224,224))
#img=image.load_img('Datasets/test/PNEUMONIA/person1951_bacteria_4882.jpeg',target_size=(224,224))
#img=image.load_img('Datasets/test/PNEUMONIA/person1952_bacteria_4883.jpeg',target_size=(224,224))

# testing for healthy images
#img=image.load_img('Datasets/test/NORMAL/NORMAL2-IM-1427-0001.jpeg',target_size=(224,224))
#img=image.load_img('Datasets/test/NORMAL/NORMAL2-IM-1430-0001.jpeg',target_size=(224,224))
#img=image.load_img('Datasets/test/NORMAL/NORMAL2-IM-1431-0001.jpeg',target_size=(224,224))
#img=image.load_img('Datasets/test/NORMAL/NORMAL2-IM-1436-0001.jpeg',target_size=(224,224))
#img=image.load_img('Datasets/test/NORMAL/NORMAL2-IM-1437-0001.jpeg',target_size=(224,224))

x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)
result=int(classes[0][0])

#prints out if patient has pneumonia or if patient is healthy based on the trained model provided
if result==0: print("Patient has PNEUMONIA")
else: print("Patient is Healthy")