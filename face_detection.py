import cv2 as cv


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

face = cv.VideoCapture(0)

while True:
    isTrue , frame = face.read()

    grayScale = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayScale, 1.1, 4)

    for (x , y , w, h) in faces :
        cv.rectangle(frame , (x,y) , (x+w,y+h) , (0,255,0) , thickness=5)

    cv.imshow('faces' , frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

face.release()
cv.destroyAllWindows()