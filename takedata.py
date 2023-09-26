import cv2
from tkinter import *
from tkinter import ttk
face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
id=int(input("enter persion ID"))
img_id=1

def facecrop(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        facecrop=img[y:y+h,x:x+w]
        return facecrop
    
cap=cv2.VideoCapture(1)
imgId=0
while True:
    ret,myFrame=cap.read()
    if facecrop(myFrame) is not None:
        imgId+=1
        face=cv2.resize(facecrop(myFrame),(600,600))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        facedataPath="faceData/user."+str(id)+"."+str(img_id)+".jpg"
        cv2.imwrite(facedataPath,face)
        cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
        cv2.imshow("live",face)
        img_id+=1
    if cv2.waitKey(1)==13 or int(img_id)==500:
        break
cap.release()


