import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("classifier.xml")


cap = cv2.VideoCapture(1)
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

        if confidence < 70:
            if id_==1:
                n ="Dhanush GN"
            elif id_==2:
                n="Jhanavi GN"
            name = f"{n}"
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
