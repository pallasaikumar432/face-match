import cv2

# Load the pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the reference image
reference_img = cv2.imread("photo.jpg")

# Check if the reference image is loaded successfully
if reference_img is None:
    print("Error: Unable to load reference image.")
    exit()

# Convert the reference image to grayscale
reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

# Check if the reference image is empty after conversion
if reference_gray is None:
    print("Error: Unable to convert reference image to grayscale.")
    exit()

# Detect faces in the reference image
reference_faces = face_cascade.detectMultiScale(reference_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        if any(x < rx + rw and x + w > rx and y < ry + rh and y + h > ry for (rx, ry, rw, rh) in reference_faces):
            color = (0, 255, 0)  # Green color for matching face
        else:
            color = (0, 0, 255)  # Red color for non-matching face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
