import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Load training images and their names
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

print("Training images:", myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:  # Ensure the image is loaded correctly
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print("Class names:", classNames)

# Function to find encodings for training images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        encode = face_recognition.face_encodings(img)  # Get face encodings
        if encode:  # Ensure an encoding is found
            encodeList.append(encode[0])
    return encodeList

# Function to mark attendance in a CSV file
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        if name not in nameList:  # Only mark attendance if not already marked
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Generate encodings for known images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:  # Check if the frame is captured successfully
        print("Failed to capture image from webcam.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize image for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    facesCurFrame = face_recognition.face_locations(imgS)  # Locate faces in the current frame
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # Get encodings for the current frame

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # Compare faces
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Calculate distances
        matchIndex = np.argmin(faceDis)  # Find the index of the closest match

        if matches[matchIndex] and faceDis[matchIndex] < 0.6:  # Threshold for recognizing a face
            name = classNames[matchIndex].upper()  # Get the name of the matched face
        else:
            name = "Unknown Person"  # If no match is found

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale up face location

        # Draw rectangles around the detected face and display the name
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        if name != "Unknown Person":
            # Mark attendance for recognized person
            markAttendance(name)

    cv2.imshow('Webcam', img)  # Display the webcam feed

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
