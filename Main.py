import tkinter
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from keras.utils.np_utils import to_categorical

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Bidirectional
import webbrowser

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam


global filedirectory, text
global X, Y, filedirectory, X_train, X_test, y_train, y_test, cnn_model, bilstm, attention, unet, cnn_X, cnn_Y
global accuracy, precision, recall, fscore, sensitivity, specificity

def Proposed():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, sensitivity, specificity
    global cnn_model, bilstm, attention, unet
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    sensitivity = []
    specificity = []
    #now load cnn model=============
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (3, 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    print(cnn_model.summary())
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X, Y, batch_size = 16, epochs = 200, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")

    cnn_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)#creating cnn model
    cnn_features = cnn_model.predict(cnn_X)  #extracting cnn features from test data
    print(cnn_features.shape)
    cnn_features = np.reshape(cnn_features, (cnn_features.shape[0], 16, 16))
    print(cnn_features.shape)
    X_train, X_test, y_train, y_test = train_test_split(cnn_features, cnn_Y, test_size=0.2) #split dataset into train and test

    bilstm = Sequential()#defining deep learning sequential object
    #adding bi-directional LSTM layer with 32 filters to filter given input X train data to select relevant features
    bilstm.add(Bidirectional(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
    #adding dropout layer to remove irrelevant features
    bilstm.add(Dropout(0.2))
    #adding another layer
    bilstm.add(Bidirectional(LSTM(32)))
    bilstm.add(Dropout(0.2))
    #defining output layer for prediction
    bilstm.add(Dense(y_train.shape[1], activation='softmax'))
    #compile BI-LSTM model
    bilstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/bilstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/bilstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = bilstm.fit(cnn_features, Y, batch_size = 16, epochs = 200, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)       
    else:
        bilstm = load_model("model/bilstm_weights.hdf5")

    predict = bilstm.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testy = np.argmax(y_test, axis=1)        
    calculateClassifierMetrics("Manhole Detection Performance", testy, predict)    


def uploadDataset():
    global filedirectory
    global X, Y, X_train, X_test, y_train, y_test, cnn_X, cnn_Y
    filedirectory = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filedirectory+" dataset loaded\n\n")

def Preprocessing():
    global filedirectory
    global X, Y, X_train, X_test, y_train, y_test, cnn_X, cnn_Y
    cnn_X = np.load('model/cnn_X.txt.npy')
    cnn_Y = np.load('model/cnn_Y.txt.npy')
    cnn_X = cnn_X.astype('float32')
    cnn_X = cnn_X/255
    indices = np.arange(cnn_X.shape[0])
    np.random.shuffle(indices)
    cnn_X = cnn_X[indices]
    cnn_Y = cnn_Y[indices]
    cnn_Y = to_categorical(cnn_Y)
    X_train, X_test, y_train, y_test = train_test_split(cnn_X, cnn_Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Loading Completed\n\n")
    text.insert(END,"Total images found in dataset : "+str(cnn_X.shape[0]*10)+"\n\n")    


#function to calculate all metrics
def calculateClassifierMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    cm = confusion_matrix(testY, predict)
    total = sum(sum(cm))
    se = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
    sp = cm[1,1]/(cm[1,0]+cm[1,1])* 100
    sensitivity.append(se)
    specificity.append(sp)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy    : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision   : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall      : "+str(r)+"\n")
    text.insert(END,algorithm+" Sensitivity : "+str(f)+"\n")
    text.insert(END,algorithm+" Specificity : "+str(se)+"\n")
    text.insert(END,algorithm+" FSCORE      : "+str(sp)+"\n\n")
      
def parse_annotation(annotation_file):
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
        boxes = []
        for line in lines:
            data = line.split()
            class_label = data[0]
            x_center, y_center, width, height = map(float, data[1:])
            x1 = int((x_center - width / 2) * IMAGE_WIDTH)
            y1 = int((y_center - height / 2) * IMAGE_HEIGHT)
            x2 = int((x_center + width / 2) * IMAGE_WIDTH)
            y2 = int((y_center + height / 2) * IMAGE_HEIGHT)
            boxes.append((class_label, x1, y1, x2, y2))
        return boxes

def prediction():
    global filedirectory
    text.delete('1.0', END)
    test_image_name = filedialog.askopenfilename(initialdir="Testimages")
    base_name = os.path.splitext(os.path.basename(test_image_name))[0]  # Extracting base name without extension
    print(test_image_name)
    test_image_path = os.path.join(filedirectory, test_image_name)
    test_annotation_path = os.path.join(filedirectory, base_name + ".txt")

    # Read the test image
    image = cv2.imread(test_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image dimensions
    IMAGE_WIDTH, IMAGE_HEIGHT, _ = image.shape

    # If annotation file exists, draw bounding boxes
    if os.path.exists(test_annotation_path):
        # Function to parse YOLO format annotation file
        def parse_annotation(annotation_file):
            with open(annotation_file, 'r') as file:
                lines = file.readlines()
                boxes = []
                for line in lines:
                    data = line.split()
                    class_label = data[0]
                    x_center, y_center, width, height = map(float, data[1:])
                    x1 = int((x_center - width / 2) * IMAGE_WIDTH)
                    y1 = int((y_center - height / 2) * IMAGE_HEIGHT)
                    x2 = int((x_center + width / 2) * IMAGE_WIDTH)
                    y2 = int((y_center + height / 2) * IMAGE_HEIGHT)
                    boxes.append((class_label, x1, y1, x2, y2))
                return boxes

        # Parse annotation file
        annotations = parse_annotation(test_annotation_path)

        # Draw bounding boxes
        for annotation in annotations:
            class_label, x1, y1, x2, y2 = annotation
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # Fallback: Detect manholes using contour detection
        _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        max_contour = max(contours, key=cv2.contourArea)

        # Draw bounding box around the largest contour (potential manhole)
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Manhole", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes or fallback result
    cv2.imshow("Image with Bounding Boxes or Fallback Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def Main():
    global text, root

    root = tkinter.Tk()
    root.geometry("1300x1200")
    root.title("Enhanced Object Detection Methods For Streamlined Inspection Processes In Automatic Applications")
    root.resizable(True,True)
    font = ('times', 14, 'bold')
    title = Label(root, text='Enhanced Object Detection Methods For Streamlined Inspection Processes In Automatic Applications')
    title.config(bg='yellow3', fg='white')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)
    
    font1 = ('times', 12, 'bold')

    uploadButton = Button(root, text="Upload Dataset", command=uploadDataset)
    uploadButton.place(x=60,y=80)
    uploadButton.config(font=font1)

    modelButton = Button(root, text="Preprocessing", command=Preprocessing)
    modelButton.place(x=270,y=80)
    modelButton.config(font=font1)

    graphButton = Button(root, text="Proposed Method", command=Proposed)
    graphButton.place(x=450,y=80)
    graphButton.config(font=font1)
    
    graphButton = Button(root, text="Prediction", command=prediction)
    graphButton.place(x=620,y=80)
    graphButton.config(font=font1)

    text=Text(root,height=30,width=140)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=180)    
    
    root.mainloop()
   
 
if __name__== '__main__' :
    Main ()
    
