from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import cv2
from keras.utils.np_utils import to_categorical

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

from sklearn.model_selection import train_test_split 
from keras.applications import ResNet50
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
main = tkinter.Tk()
main.title("Transfer Learning for an Automated Detection System of Fractures in Patients with Maxillofacial Trauma")
main.geometry("1200x1200")
global X_train, X_test, y_train, y_test
global model
global filename
global X, Y
global accuracy,precision,recall,fscore
labels = ['Fracture','Nofracture']
def getLabel(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index
    def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")
def preprocessDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j]) #read image from dataset directory
                    img = cv2.resize(img, (64,64)) #resize image
                    im2arr = np.array(img)


                    im2arr = im2arr.reshape(64,64,3) #image as 3 colour format
                   X.append(im2arr) #add images to array
                    label = getLabel(name)
                    Y.append(label) #add class label to Y variable
                    print(name+" "+directory[j]+" "+str(label))
        X = np.asarray(X) #convert array images to numpy array
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)

    X = X.astype('float32')
    X = X/255 #normalize image
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle images data
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and tesrt        
    
    
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Different categories found in dataset\n\n")
    text.insert(END,str(labels)+"\n\n")
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"Total images used to train ShrimpNet : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total images used to test ShrimpNet  : "+str(X_test.shape[0])+"\n")

    test = X[300]
    test = cv2.resize(test,(300,300))
    cv2.imshow("Sample Processes Image",test)
    cv2.waitKey(0)


def trainResnet():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    global model
    #create resnet50 object
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in resnet.layers:
        layer.trainable = False
    #create own CNN Model object    
    classifier = Sequential()
    #add resnet50 to our model as transfer leanring

    classifier.add(resnet)
    #adding CNN layer with 32 filters as input
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    #creating CNN output layer for prediction
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])    
    with open('model/resnet_model.json', "r") as json_file: 
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    json_file.close()
    model.load_weights("model/resnet_model_weights.h5") #MNIST model will be loaded here
    model._make_predict_function()
    
    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    for i in range(0,5):
        predict[i] = 0
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    
    text.insert(END,'Resnet Transfer Learning Accuracy  : '+str(a)+"\n")
    text.insert(END,'Resnet Transfer Learning Precision : '+str(p)+"\n")
    text.insert(END,'Resnet Transfer Learning Recall    : '+str(r)+"\n")
    text.insert(END,'Resnet Transfer Learning FMeasure  : '+str(f)+"\n")
    LABELS = labels
    cm = confusion_matrix(y_test, predict)
    
    plt.figure(figsize =(8, 6)) 
    ax = sns.heatmap(cm, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Resnet Transfer Learning Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()


        
def predict():
    text.delete('1.0', END)
    global model
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = model.predict(img)
    predict = np.argmax(preds)
    print(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Image Predicted as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Image Predicted as : '+labels[predict], img)
    cv2.waitKey(0)



def graph():
    f = open('model/resnet_history.pckl', 'rb')
    fracture = pickle.load(f)
    f.close()
    accuracy = fracture['accuracy']
    error = fracture['loss']

    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Error Rate')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(error, 'ro-', color = 'blue')
    plt.legend(['Resnet50 Transfer Learning Accuracy', 'Resnet50 Transfer Learning Loss'])
    plt.title('Resnet50 Transfer Learning Accuracy & Error Graph')
    plt.show()

def close():
    main.destroy()

font = ('times', 14, 'bold')

title = Label(main, text='Transfer Learning for an Automated Detection System of Fractures in Patients with Maxillofacial Trauma')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Skull-Fracture Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

hybridMLButton = Button(main, text="Train Resnet50 CNN Model", command=trainResnet)
hybridMLButton.place(x=50,y=200)
hybridMLButton.config(font=font1)

snButton = Button(main, text="Accuracy Graph", command=graph)
snButton.place(x=50,y=250)
snButton.config(font=font1)

snButton = Button(main, text="Predict Fracture from Test Image", command=predict)
snButton.place(x=50,y=300)
snButton.config(font=font1)

graphButton = Button(main, text="Exit", command=close)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)
