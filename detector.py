import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml" 
faceDetect = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor =(255,255,255)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(Id==1):
            Id="SIMRANJIT SINGH, AGE-22"
        elif(Id==5):
            Id="Munish"
        else:
            Id="Unknown"
        cv2.putText(im,str(Id),(x,y+h),fontface,fontscale,fontcolor)
    cv2.imshow('im',im) 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
