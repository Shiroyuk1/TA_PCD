''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import pandas as pd
import os 
import datetime
from openpyxl import load_workbook

def write_to_csv(id, confidence, names):
    if ((confidence < 80) and (confidence >= 0)):
        id = names[id]
        confidence = "  {0}%".format(round(100 - confidence))

        # membaca file CSV
        try:
            df = pd.read_csv("absensi.csv")
        except:
            df = pd.DataFrame(columns=['Nama', 'Tanggal Absensi'])
        
        # memeriksa apakah nama sudah ada dalam file CSV
        name_exists = (df['Nama'] == id).any()
        
        # menambahkan atau memperbarui tanggal absensi
        if name_exists:
            index = df[df['Nama'] == id].index[0]
            df.at[index, 'Tanggal Absensi'].append(datetime.datetime.now())
        else:
            data = {'Nama': id, 'Tanggal Absensi': [datetime.datetime.now()]}
            df = df.append(pd.DataFrame(data), ignore_index=True)
        
        df.to_csv("absensi.csv", index = False, mode = 'w')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Izzaqi', 'Ejre', 'Fahre', 'Tedy', 'Nabil'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
   
        
        # if ((confidence < 80) and (confidence >= 0 ) ):
        #     id = names[id]
        #     confidence = "  {0}%".format(round(100 - confidence))

        #     data = {'Nama': id, 'Tanggal Absensi': [datetime.datetime.now()] }
        #     df = pd.DataFrame(data)
        #     df.to_csv("absensi.csv", index = False, mode = 'a')
        if ((confidence < 80) and (confidence >= 0 ) ):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))

            try:
                df = pd.read_csv("absensi.csv")
                mask = df['Nama'] == id
                if mask.any():
                    df.loc[mask, 'Tanggal Absensi'].iloc[-1] = datetime.datetime.now()
                else:
                    data = {'Nama': id, 'Tanggal Absensi': [datetime.datetime.now()] }
                    df = df.append(pd.DataFrame(data), ignore_index=True)
                df.to_csv("absensi.csv", index = False)
                
            except FileNotFoundError:
                # buat file baru dan tambahkan kolom 'Nama' dan 'Tanggal Absensi'
                data = {'Nama': id, 'Tanggal Absensi': [datetime.datetime.now()] }
                df = pd.DataFrame(data)
                df.to_csv("absensi.csv", index = False)

        elif(confidence >= 20):
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

       

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
