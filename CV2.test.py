import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('./openCV/haarcascade_frontalface_alt2.xml')

while True:
    ret, img= cap.read()
    
    img = cv2.flip(img,2)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_frame = (0,0,0)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),face_frame,2)
        cv2.putText(img, "X = "+ str(x),(0,20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,cv2.LINE_AA)
        cv2.putText(img, "Y = "+ str(y),(0,40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,cv2.LINE_AA)
        cv2.putText(img, "W = "+ str(w),(0,60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,cv2.LINE_AA)
        cv2.putText(img, "H = "+ str(h),(0,80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,cv2.LINE_AA)
        
    cv2.imshow('Face_Detect_Test', img)
    
    k = cv2.waitKey(1)
    
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()