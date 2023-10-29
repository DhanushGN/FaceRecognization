from tkinter import *
from tkinter import simpledialog
from tkinter import messagebox
from PIL import Image,ImageTk
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
import cv2

def reco():
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("classifier.xml")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:

            face_roi = gray[y:y + h, x:x + w]

            id_ , confidence = recognizer.predict(face_roi)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            

            if confidence<60:
                f=open("name.txt","r")
                content=f.readlines()
                n =content[id_-1]
                name=n[4:]
                confidence_text = f"Confidence: {round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"Confidence: {round(100 - confidence)}%"
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def addface():
        face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        ROOT=Tk()
        ROOT.withdraw()
        id=simpledialog.askinteger(title="USER ID",prompt="Enter Your ID")
        Name=simpledialog.askstring(title="USER ID",prompt="Enter Your ID")
        f=open("name.txt","a")
        f.write(f"{id}st {Name}\n")
        f.close
        img_id=1

        def facecrop(img):
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                facecrop=img[y:y+h,x:x+w]
                return facecrop
            
        cap=cv2.VideoCapture(0)
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
            if cv2.waitKey(1)==13 or int(img_id)==5000:
                break
        cap.release()

def traindata():
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
        messagebox.showinfo("ALLERT","Training is Done")
    data_dir="faceData"
    trainData(data_dir)

def mark_attendance():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("classifier.xml")
    attendance_data = {} 
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            id_, confidence = recognizer.predict(face_roi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if confidence < 60:
                f = open("name.txt", "r")
                content = f.readlines()
                n = content[id_ - 1]
                name = n[4:]
                confidence_text = f"Confidence: {round(100 - confidence)}%"
                if name not in attendance_data:
                    attendance_data[name] = "Present"
            else:
                name = "Unknown"
                confidence_text = f"Confidence: {round(100 - confidence)}%"
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Face Recognition', frame)
        root = ET.Element("attendance")
        for name, status in attendance_data.items():
                student = ET.SubElement(root, "student")
                student.set("name", name)
                student.set("status", status)
        tree = ET.ElementTree(root)
        tree.write("attendance.xml")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def attend_button_click():
    mark_attendance()



root=Tk()

root.title("FACE RECOGNIZATION")
root.geometry('1900x1000')

#background img------------------------------
bg_img=Image.open("800_390_face-recognition.png")
bg_img=bg_img.resize((1900,1000))
main_img=ImageTk.PhotoImage(bg_img)
main_lbl=Label(root,image=main_img)
main_lbl.place(x=0,y=0)


title_txt=Label(root,text="Face Recognization",font=("times new roman",50,"bold"),)
title_txt.place(x=650,y=10)

StudentName_txt=Label(root,text="Dhanush GN \n Seshadripuram Degree College  \n BCA 3rd Year",font=("times new roman",30,"bold"),)
StudentName_txt.place(x=1300,y=700)

addface_txt=Label(root,text="ADD FACE",font=("times new roman",25,"bold"),)
addface_txt.place(x=165,y=150)
addface_img=Image.open("1_YF4KdQE-RadFtNa6n66wdg.gif")
addface_img=addface_img.resize((300,300))
mainaddface_img=ImageTk.PhotoImage(addface_img)
addface_btn=Button(root,image=mainaddface_img,command=addface)
addface_btn.place(x=100,y=200,height=300,width=300)

title_txt=Label(root,text="TRAIN DATA",font=("times new roman",25,"bold"),)
title_txt.place(x=580,y=150)
train_img=Image.open("trainface.jpg")
train_img=train_img.resize((300,300))
maintrain_img=ImageTk.PhotoImage(train_img)
mainyrain_btn=Button(root,image=maintrain_img,command=traindata)
mainyrain_btn.place(x=550,y=200,height=300,width=300)

title_txt=Label(root,text="RECOGNIZE FACES",font=("times new roman",25,"bold"),)
title_txt.place(x=90,y=550)
recognizeface_img=Image.open("Telpo-TPS980-face-recognition-machine.jpg")
recognizeface_img=recognizeface_img.resize((300,300))
maintrainface_img=ImageTk.PhotoImage(recognizeface_img)
mainrecon_btn=Button(root,image=maintrainface_img,command=reco)
mainrecon_btn.place(x=100,y=600,height=300,width=300)

title_txt=Label(root,text="ATTENDANCE",font=("times new roman",25,"bold"),)
title_txt.place(x=580,y=550)
attendance_img=Image.open("Screenshot 2023-10-01 122705.png")
attendance_img=attendance_img.resize((300,300))
mainatted_img=ImageTk.PhotoImage(attendance_img)
attend_btn=Button(root,image=mainatted_img,command=attend_button_click)
attend_btn.place(x=550,y=600,height=300,width=300)

root.mainloop()

