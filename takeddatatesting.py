from PIL import Image,ImageTk
import numpy as np
import os
import cv2

def trainData(data_dir):
    path=[os.path.join(data_dir,file) for file in os.listdir(data_dir)]
    faces=[]
    ids=[]
    for image in path:
        z=image[14:]
        img=Image.open(image).convert('L')
        img_np=np.array(img,'uint8')
        id=int(z[0])
        faces.append(img_np)
        ids.append(id)
    ids=np.array(ids)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,ids)
    recognizer.write("classifier.xml")
    print("traning done")
data_dir="faceData"
trainData(data_dir)