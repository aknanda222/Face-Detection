import cv2
from random import randrange

# To load some pre-trained data on face frontals from OpenCV
trained_face_data = cv2.CascadeClassifier('haas_cascade.xml')

#Choose an image to detect faces in
img = cv2.imread('rd1.jpg')

#To capture video from webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 1280)
webcam.set(4, 720)
webcam.set(10, 100)
while True:
    #To get the frame from the Webcam
    success_frame_read, frame = webcam.read()
    #Must convert it into grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    #To draw rectangles around the face
    for (x, y, w, h) in face_coordinates:
       cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)

    #To show the face   
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release the VideoCapture Object
webcam.release()