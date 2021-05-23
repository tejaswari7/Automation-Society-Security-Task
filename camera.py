import cv2
import os
import time 
import numpy as np
import keras
import threading
import datetime
import pyrebase
import pyttsx3 as p
import webbrowser 
import os
import shutil
from os import path
from scipy import stats
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model,model_from_json
from flask import request



class VideoCamera(object):
    global speak
    def __init__(self):
       #capturing video
        self.video = cv2.VideoCapture(0)
        config = {
                "Enter firebase details here"
              };
           # Initialize Firebase
        firebase=pyrebase.initialize_app(config)
        self.db = firebase.database()
        self.storage = firebase.storage()
    
    path = "Models/model.h5"
    global enc_model,mtcnn_detector,category2label,colors,model
    enc_model = load_model(path) 
    model = load_model("Models/inceptionV3-model.h5")    
    category2label = {0: 'without_mask', 1: 'with_mask', 2: 'mask_weared_incorrect'}
    colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (0, 255, 255)}
    # print("\n\n1:model loaded\n\n")

    def __del__(self):
        #releasing camera
        self.video.release()

    mtcnn_detector = MTCNN()
    def detect_face(filename, required_size=(160, 160),normalize = True):
        # print("\n\ninside detect_face\n\n")
        # mtcnn_detector = MTCNN()
        img = Image.open(filename)
        img = img.convert('RGB')
        pixels = np.asarray(img)
        global results
        results = mtcnn_detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
    
        if normalize == True:

            mean = np.mean(face_array, axis=(0,1,2), keepdims=True)
            std = np.std(face_array, axis=(0,1,2), keepdims=True)
            std_adj = np.maximum(std, 1.0)
            return (face_array - mean) / std

        else : 
            return face_array

    global known_faces_encodings
    global known_faces_ids 
    known_faces_encodings = []
    known_faces_ids = []
    
    known_faces_path = "Face_database/"  #Store your image by creating Face_database folder

    for filename in os.listdir(known_faces_path):
        
        # Detect faces
        face = detect_face(known_faces_path+filename,normalize = True)

        # Compute face encodings

        feature_vector = enc_model.predict(face.reshape(1,160,160,3))
        feature_vector/= np.sqrt(np.sum(feature_vector**2))
        known_faces_encodings.append(feature_vector)

        # Save Person IDs
        label = filename.split('.')[0]
        known_faces_ids.append(label)


    known_faces_encodings = np.array(known_faces_encodings).reshape(len(known_faces_encodings),128)
    known_faces_ids = np.array(known_faces_ids)
    print("\n\n2:known_face\n\n")

    def recognize(self,img,known_faces_encodings,known_faces_ids,threshold = 0.75):
        scores = np.zeros((len(known_faces_ids),1),dtype=float)

        enc = enc_model.predict(img.reshape(1,160,160,3))
        enc/= np.sqrt(np.sum(enc**2))

        scores = np.sqrt(np.sum((enc-known_faces_encodings)**2,axis=1))

        match = np.argmin(scores)
        if scores[match] > threshold :
            return ("UNKNOWN",0)
            
        else :
            return (known_faces_ids[match],scores[match])
    

    def get_frame(self):
        prev_name = "UNKNOWN"
        ret, img = self.video.read()
        global flat,name,ct,t,icount,uname
        name,ct,icount,flat = '','',1,0
        print("executed name and ct")
        
        results = mtcnn_detector.detect_faces(img)
        # if(len(results)==0):
        #   continue
        print("results ",len(results))
        faces = []
        for i in range(len(results)):
            x,y,w,h = results[i]['box']
            x, y = abs(x), abs(y)
            faces.append([x,y,w,h])
        
        for (x, y, w, h) in faces:
            image = Image.fromarray(img[y:y+h, x:x+w])
            image = image.resize((160,160))
            face_array = asarray(image)

            roi =  img[y:y+h, x:x+w]
            data = cv2.resize(roi,(100,100))
            data = data / 255.
            data = data.reshape((1,) + data.shape)
            scores = model.predict(data)
            target = np.argmax(scores, axis=1)[0]
            print("in for")

            if category2label[target] == 'with_mask':
                cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=colors[target], thickness=2)
                text = "Please Remove Mask"
                p.speak(text)
                print("below text")
                cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                print("continue")
                continue
            print("after mask")
            mean = np.mean(face_array, axis=(0,1,2), keepdims=True)
            std = np.std(face_array, axis=(0,1,2), keepdims=True)
            std_adj = np.maximum(std, 1.0)
            face_array_normalized = (face_array - mean) / std
            label = self.recognize(face_array_normalized,known_faces_encodings,known_faces_ids,threshold = 0.75)
            ct = str(datetime.datetime.now())
            name = label[0]
            print(name)
            
                
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(img, label[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            last_record = self.db.child("EOD").order_by_key().limit_to_last(1).get().val()
            # print(last_record)
            for i, v in last_record.items():
                prev_name = v["Name"]

            if name != "UNKNOWN" and name != prev_name:
                data = { 'Timestamp':ct,'Name': name, 'Temperature':t,'Category':'Resident'}
                self.db.child("EOD").push(data)
                self.db.child("Temp").set(float(t))
                prev_name = name

            elif (name == "UNKNOWN"):
                cv2.imwrite("Unknown/" + "Unknown" + ".jpg", img[y:y+h,x:x+w])
                # # cv2.imshow('image', img)
                
                webbrowser.open('http://Enter flask server url/audio')
                time.sleep(35)
                img_name = uname+'.jpg'
                source = 'Unknown/Unknown.jpg'
                dest = 'Unknown/' + img_name
                os.rename(source, dest)
                self.storage.child("Unknown/"+ uname + ".jpg").put("Unknown/" + uname + ".jpg")
                link = self.storage.child("Unknown/"+ uname + ".jpg").get_url(None)
                data = {'Timestamp':ct,'Name': uname, 'Temperature':t, 'Img_Link':link,'Category':'Visitor', 'Flat_Id': flat}
                print(data)
                self.db.child("EOD").push(data)
                self.db.child("Request").update(data)
                
                
                icount = icount + 1
                # print("u r UNKNOWN")
                print("icount= ",icount)
                # self.db.child("Mic").set(0)
                
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()