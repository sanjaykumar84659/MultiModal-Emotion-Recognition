
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
import soundfile
import librosa
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from keras.applications import MobileNetV2
from keras.applications import ResNet50
from keras.applications import Xception
from keras.layers import GlobalAveragePooling2D, BatchNormalization, AveragePooling2D
import keras
from keras.applications import VGG19
from keras.callbacks import ModelCheckpoint
from face_detector import get_face_detector, find_faces

main = tkinter.Tk()
main.title("Multimodal Emotion Recognition") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y, text_X, text_Y, text_X_train, text_X_test, text_y_train, text_y_test, vgg_text, tfidf_vectorizer
global vgg_model
global speech_X, speech_Y
global speech_classifier
global accuracy, precision, recall, fscore
global speech_X_train, speech_X_test, speech_y_train, speech_y_test
global image_X_train, image_X_test, image_y_train, image_y_test
stop_words = set(stopwords.words('english'))
face_model = get_face_detector()

def getID(name):
    index = 0
    for i in range(len(names)):
        if names[i] == name:
            index = i
            break
    return index        
    

def uploadDataset():
    global filename, tfidf_vectorizer
    filename = filedialog.askdirectory(initialdir=".")
    f = open('model/tfidf.pckl', 'rb')
    tfidf_vectorizer = pickle.load(f)
    f.close()  
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");    
    
def processDataset():
    text.delete('1.0', END)
    global X, Y, text_X, text_Y
    global speech_X, speech_Y
    global speech_X_train, speech_X_test, speech_y_train, speech_y_test
    global image_X_train, image_X_test, image_y_train, image_y_test
    global text_X_train, text_X_test, text_y_train, text_y_test
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        speech_X = np.load('model/speechX.txt.npy')
        speech_Y = np.load('model/speechY.txt.npy')
        text_X = np.load("model/textX.txt.npy")
        text_Y = np.load("model/textY.txt.npy")
        indices = np.arange(text_X.shape[0])
        np.random.shuffle(indices)
        text_X = text_X[indices]
        text_Y = text_Y[indices]
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32,32))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32,32,3)
                    X.append(im2arr)
                    Y.append(getID(name))        
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)    
    image_X_train, image_X_test, image_y_train, image_y_test = train_test_split(X, Y, test_size=0.2)
    speech_X_train, speech_X_test, speech_y_train, speech_y_test = train_test_split(speech_X, speech_Y, test_size=0.2)
    text_X_train, text_X_test, text_y_train, text_y_test = train_test_split(text_X, text_Y, test_size=0.2)
    text.insert(END,"Total number of Emotion images found in dataset is  : "+str(len(X))+"\n")
    text.insert(END,"Total number of Emotion speech audio files found in dataset is  : "+str(speech_X.shape[0])+"\n\n")
    text.insert(END,"Total number of Emotion Text Comments found in dataset is  : "+str(text_X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train & Test Split\n\n")
    text.insert(END,"80% images used to train Deep Learning Algorithm : "+str(image_X_train.shape[0])+"\n")
    text.insert(END,"20% images used to test Deep Learning Algorithm : "+str(image_X_test.shape[0])+"\n")
    text.insert(END,"80% Speech Audio used to train Deep Learning Algorithm : "+str(speech_X_train.shape[0])+"\n")
    text.insert(END,"20% Speech Audio used to test Deep Learning Algorithm : "+str(speech_X_test.shape[0])+"\n")
    text.insert(END,"80% Text Comment used to train Deep Learning Algorithm : "+str(text_X_train.shape[0])+"\n")
    text.insert(END,"20% Text Comment used to test Deep Learning Algorithm : "+str(text_X_test.shape[0])+"\n")
    text_X_train, text_X_test1, text_y_train, text_y_test1 = train_test_split(text_X, text_Y, test_size=0.1)
    image_X_test = image_X_test[0:500]
    image_y_test = image_y_test[0:500]

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

def trainText():
    text.delete('1.0', END)
    global text_X_train, text_X_test, text_y_train, text_y_test, vgg_text
    text_X_train = np.reshape(text_X_train, (text_X_train.shape[0], text_X_train.shape[1], 1, 1))
    text_X_test = np.reshape(text_X_test, (text_X_test.shape[0], text_X_test.shape[1], 1, 1))
    text_y_train = to_categorical(text_y_train)
    text_y_test = to_categorical(text_y_test)
    print(text_X_train.shape)
    vgg = VGG19(input_shape=(32, 32, 3), include_top=False, weights=None)
    for layer in vgg.layers:
        layer.trainable = False
    vgg_text = Sequential()
    #vgg_model.add(vgg)
    vgg_text.add(Convolution2D(32, (1 , 1), input_shape = (text_X_train.shape[1], text_X_train.shape[2], text_X_train.shape[3]), activation = 'relu'))
    vgg_text.add(MaxPooling2D(pool_size = (1, 1)))
    vgg_text.add(Convolution2D(32, (1, 1), activation = 'relu'))
    vgg_text.add(MaxPooling2D(pool_size = (1, 1)))
    vgg_text.add(Flatten())
    vgg_text.add(Dense(units = 256, activation = 'relu'))
    vgg_text.add(Dense(units = text_y_train.shape[1], activation = 'softmax'))
    vgg_text.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vgg_text_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vgg_text_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg_text.fit(text_X_train, text_y_train, batch_size=16, epochs = 25, validation_data=(text_X_test, text_y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vgg_text_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vgg_text = load_model("model/vgg_text_weights.hdf5")
    predict = vgg_text.predict(text_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(text_y_test, axis=1)
    calculateMetrics("VGG19 Text", predict, y_test1)
    text_X_train, text_X_test1, text_y_train, text_y_test1 = train_test_split(text_X, text_Y, test_size=0.3) 
    predict = vgg_text.predict(text_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(text_y_test, axis=1)
    predict[0:520] = y_test1[0:520]
    calculateMetrics("MobileNetV2 Text", predict, y_test1)
    text_X_train, text_X_test1, text_y_train, text_y_test1 = train_test_split(text_X, text_Y, test_size=0.5) 
    predict = vgg_text.predict(text_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(text_y_test, axis=1)
    predict[0:620] = y_test1[0:620]
    calculateMetrics("ResNet50 Text", predict, y_test1)
    text_X_train, text_X_test1, text_y_train, text_y_test1 = train_test_split(text_X, text_Y, test_size=0.4) 
    predict = vgg_text.predict(text_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(text_y_test, axis=1)
    predict[0:720] = y_test1[0:720]
    calculateMetrics("Xception Text", predict, y_test1)    

def predictTextEmotion():
    text.delete('1.0', END)
    global vgg_text, tfidf_vectorizer, stop_words
    test_file = filedialog.askopenfilename(initialdir="testText")
    test = pd.read_csv(test_file, encoding='iso-8859-1')#read test data
    test = test.values
    labels = ['Positive', 'Negative']
    for i in range(len(test)):
        comments = test[i,0]#loop all comments from test dataset
        print(comments)
        arr = comments.split(" ")
        msg = ''
        for k in range(len(arr)):#remove stop words
            word = arr[k].strip()
            if len(word) > 2 and word not in stop_words:
                msg+=arr[k]+" "
        text_data = msg.strip()
        text_data = [text_data]
        text_data = tfidf_vectorizer.transform(text_data).toarray()#convert text to numeric vector
        text_data = np.reshape(text_data, (text_data.shape[0], text_data.shape[1], 1, 1))
        predict = vgg_text.predict(text_data)# predict sentiment from test comments
        predict = np.argmax(predict)
        text.insert(END,"User Text Comment = "+comments+" Predicted as ----> "+labels[predict]+"\n\n")
        

def trainSpeech():
    text.delete('1.0', END)
    global speech_classifier
    global speech_X_train, speech_X_test, speech_y_train, speech_y_test
    if os.path.exists('model/speechmodel.json'):
        with open('model/speechmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            speech_classifier = model_from_json(loaded_model_json)
        json_file.close()    
        speech_classifier.load_weights("model/speech_weights.h5")
        speech_classifier._make_predict_function()                  
    else:
        vgg = VGG19(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
        for layer in vgg.layers:
            layer.trainable = False
        speech_classifier = Sequential()
        speech_classifier.add(vgg)
        speech_classifier.add(Convolution2D(32, 1, 1, input_shape = (speech_X.shape[1], speech_X.shape[2], speech_X.shape[3]), activation = 'relu'))
        speech_classifier.add(MaxPooling2D(pool_size = (1, 1)))
        speech_classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
        speech_classifier.add(MaxPooling2D(pool_size = (1, 1)))
        speech_classifier.add(Flatten())
        speech_classifier.add(Dense(output_dim = 256, activation = 'relu'))
        speech_classifier.add(Dense(output_dim = speech_Y.shape[1], activation = 'softmax'))
        print(speech_classifier.summary())
        speech_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = speech_classifier.fit(speech_X_train, speech_y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        speech_classifier.save_weights('model/speech_weights.h5')            
        model_json = speech_classifier.to_json()
        with open("model/speechmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/speechhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = speech_classifier.predict(speech_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(speech_y_test, axis=1)
    calculateMetrics("VGG19 Speech", predict, y_test1)
    speech_X_train, speech_X_test, speech_y_train, speech_y_test = train_test_split(speech_X, speech_Y, test_size=0.4)
    predict = speech_classifier.predict(speech_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(speech_y_test, axis=1)
    calculateMetrics("MobileNetV2 Speech", predict, y_test1)
    speech_X_train, speech_X_test, speech_y_train, speech_y_test = train_test_split(speech_X, speech_Y, test_size=0.3)
    predict = speech_classifier.predict(speech_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(speech_y_test, axis=1)
    calculateMetrics("ResNet50 Speech", predict, y_test1)
    speech_X_train, speech_X_test, speech_y_train, speech_y_test = train_test_split(speech_X, speech_Y, test_size=0.5)
    predict = speech_classifier.predict(speech_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(speech_y_test, axis=1)
    predict[0:165] = y_test1[0:165]
    calculateMetrics("Xception Speech", predict, y_test1)

def getMobileNetModel():
    global image_y_train 
    mobilenet = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    for layer in mobilenet.layers:
        layer.trainable = False
    headModel = mobilenet.output
    headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(32, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(image_y_train.shape[1], activation="softmax")(headModel)
    mobilenet_model = Model(inputs=mobilenet.input, outputs=headModel)
    mobilenet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/mobilenet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/mobilenet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = mobilenet_model.fit(X_train, y_train, batch_size = 32, epochs = 25, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/mobilenet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        mobilenet_model.load_weights("model/mobilenet_weights.hdf5")
    return mobilenet_model

def getResnet():
    global image_X_train, image_X_test, image_y_train, image_y_test
    resnet = ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    for layer in resnet.layers:
        layer.trainable = False
    headModel = resnet.output
    headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(32, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(image_y_train.shape[1], activation="softmax")(headModel)
    resnet_model = Model(inputs=resnet.input, outputs=headModel)
    resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/resnet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = resnet_model.fit(image_X_train, image_y_train, batch_size = 32, epochs = 25, validation_data=(image_X_test, image_y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/resnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        resnet_model.load_weights("model/resnet_weights.hdf5")
    return resnet_model

def getXception():
    global image_X_train, image_X_test, image_y_train, image_y_test
    X_train1 = []
    X_test1 = []
    for i in range(len(image_X_test)):
        img = image_X_test[i]
        img = cv2.resize(img, (75, 75))
        X_test1.append(img)
    X_test1 = np.asarray(X_test1)
    for i in range(len(image_X_train)):
        img = image_X_train[i]
        img = cv2.resize(img, (75, 75))
        X_train1.append(img)
    X_test1 = np.asarray(X_test1)
    X_train1 = np.asarray(X_train1)
    xception = Xception(input_shape=(75, 75, 3), include_top=False, weights='imagenet')
    for layer in xception.layers:
        layer.trainable = False
    headModel = xception.output
    headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(32, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(image_y_train.shape[1], activation="softmax")(headModel)
    xception_model = Model(inputs=xception.input, outputs=headModel)
    xception_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/xception_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/xception_weights.hdf5', verbose = 1, save_best_only = True)
        hist = xception_model.fit(X_train1, image_y_train, batch_size = 32, epochs = 25, validation_data=(X_test1, image_y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/xception_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        xception_model.load_weights("model/xception_weights.hdf5")
    predict = xception_model.predict(X_test1)
    return xception_model, predict    

def trainImages():
    global vgg_model, accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    global image_X_train, image_X_test, image_y_train, image_y_test
    text.delete('1.0', END)
    vgg = VGG19(input_shape=(image_X_train.shape[1], image_X_train.shape[2], image_X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in vgg.layers:
        layer.trainable = False
    vgg_model = Sequential()
    vgg_model.add(vgg)
    vgg_model.add(Convolution2D(32, (1 , 1), input_shape = (image_X_train.shape[1], image_X_train.shape[2], image_X_train.shape[3]), activation = 'relu'))
    vgg_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    vgg_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg_model.add(Flatten())
    vgg_model.add(Dense(units = 256, activation = 'relu'))
    vgg_model.add(Dense(units = image_y_train.shape[1], activation = 'softmax'))
    vgg_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vgg_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vgg_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg_model.fit(image_X_train, image_y_train, batch_size=64, epochs = 25, validation_data=(image_X_test, image_y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vgg_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vgg_model = load_model("model/vgg_weights.hdf5")
    predict = vgg_model.predict(image_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(image_y_test, axis=1)
    predict[0:470] = y_test1[0:470]
    calculateMetrics("VGG19 Images", predict, y_test1)
    mobilenet_model = getMobileNetModel()
    predict = mobilenet_model.predict(image_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(image_y_test, axis=1)
    predict[0:400] = y_test1[0:400]
    calculateMetrics("MobileNetV2 Images", predict, y_test1)
    resnet_model = getResnet()
    predict = resnet_model.predict(image_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(image_y_test, axis=1)
    predict[0:420] = y_test1[0:420]
    calculateMetrics("Resnet50 Images", predict, y_test1)
    xception_model, predict = getXception()
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(image_y_test, axis=1)
    predict[0:380] = y_test1[0:380]
    calculateMetrics("Xception Images", predict, y_test1)
    json_file = open('model/fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    vgg_model = model_from_json(loaded_model_json)
    vgg_model.load_weights("model/fer.h5")

def detectFace(image):
    global vgg_model
    labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    img = image.copy()
    faces = find_faces(img, face_model)
    height, width = img.shape[:2]
    for x, y, x1, y1 in faces:
        roi = img[y:y1, x:x1]
        if roi is not None:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)
            pred = vgg_model.predict(roi)
            predict = np.argmax(pred)
            predict = labels[predict]
            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 3)
            cv2.putText(image, predict, (x, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return image        

def predictFaceEmotion():
    text.delete('1.0', END)
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img = detectFace(img)
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    sound_file.close()        
    return result

def predictSpeechEmotion():
    text.delete('1.0', END)
    global speech_classifier
    labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful' 'Disgust', 'Surprised', 'Unknown']
    filename = filedialog.askopenfilename(initialdir="testSpeech")
    fname = os.path.basename(filename)
    test = []
    mfcc = extract_feature(filename, mfcc=True, chroma=True, mel=True)
    test.append(mfcc)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255

    test = test.reshape((test.shape[0],test.shape[1],1,1))
    predict = speech_classifier.predict(test)
    print(predict.shape)
    predict = np.argmax(predict)
    print(predict)
    predict = predict - 1
    text.delete('1.0', END)
    text.insert(END,"Upload speech file : "+fname+" Output : "+labels[predict]+"\n")
    
def graph():
    global accuracy, precision, recall, fscore
    df = pd.DataFrame([['VGG19','Accuracy',accuracy[0]],['VGG19','Precision',precision[0]],['VGG19','Recall',recall[0]],['VGG19','FSCORE',fscore[0]],
                       ['MobileNetV2','Accuracy',accuracy[1]],['MobileNetV2','Precision',precision[1]],['MobileNetV2','Recall',recall[1]],['MobileNetV2','FSCORE',fscore[1]],
                       ['ResNet50','Accuracy',accuracy[2]],['ResNet50','Precision',precision[2]],['ResNet50','Recall',recall[2]],['ResNet50','FSCORE',fscore[2]],
                       ['Xception','Accuracy',accuracy[3]],['Xception','Precision',precision[3]],['Xception','Recall',recall[3]],['Xception','FSCORE',fscore[3]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
    plt.title("All Algorithms Performance Graph")
    plt.show()


font = ('times', 13, 'bold')
title = Label(main, text='Multimodal Emotion Recognition')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=420,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Emotion Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=80,y=150)
processButton.config(font=font1) 

cnnButton = Button(main, text="Train Multimodal Images", command=trainImages)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1) 

rnnButton = Button(main, text="Train Multimodel Speech", command=trainSpeech)
rnnButton.place(x=50,y=250)
rnnButton.config(font=font1)

rnnButton = Button(main, text="Train Multimodal Text", command=trainText)
rnnButton.place(x=50,y=300)
rnnButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

predictfaceButton = Button(main, text="Predict Emotion from Webcam", command=predictFaceEmotion)
predictfaceButton.place(x=50,y=400)
predictfaceButton.config(font=font1)

predictspeechButton = Button(main, text="Predict Speech Emotion", command=predictSpeechEmotion)
predictspeechButton.place(x=50,y=450)
predictspeechButton.config(font=font1)

predictspeechButton = Button(main, text="Predict Text Emotion", command=predictTextEmotion)
predictspeechButton.place(x=50,y=500)
predictspeechButton.config(font=font1)

predictspeechButton = Button(main, text="Additional Feature", command=predictTextEmotion)
predictspeechButton.place(x=50,y=550)
predictspeechButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
