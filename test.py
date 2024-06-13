import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3


# Initialize video capture, hand detector, and classifier
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # frame rate to 30 frames per second
detector = HandDetector(maxHands=1)
classifier = Classifier("Model\keras_model.h5", "Model\labels.txt")

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Define parameters
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
sentence = ""
speaking = False

while True:
    try:
        # Capture frame from the camera
        success, img = cap.read()
        if not success:
            raise ValueError("Could not read frame")

        # Find hands in the frame
        hands, img = detector.findHands(img)

        # Display the detected hand and recognized letter
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w

            # Resize and place the cropped hand region on a white background
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get the prediction for the detected hand gesture
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            letter = labels[index]

            # Display the recognized letter on the frame
            cv2.rectangle(img, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, letter, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
        else:
            # No hand detected message
            cv2.putText(img, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Add the currently recognized sentence to the text overlay
        lines = [sentence[i:i+30] for i in range(0, len(sentence), 30)]
        textOverlay = np.ones((150 + len(lines) * 30, img.shape[1], 3), dtype=np.uint8) * 255
        for i, line in enumerate(lines):
            cv2.putText(textOverlay, line, (50, 100 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display the camera frame and text overlay
        cv2.imshow("Image", img)
        cv2.imshow("TextOverlay", textOverlay)

        # Speak the detected words after the sentence is complete
        if speaking and sentence:
            engine.say(sentence)
            engine.runAndWait()
            speaking = False

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord("s") and letter:
            sentence += letter
            print(f"Added letter: {letter}")
        elif key == ord("d"):
            sentence += " "
            print("Added space")
        elif key == ord("f"):
            speaking = True
            print("Speaking sentence:", sentence)
        elif key == ord("a"):
            if sentence:
                sentence = sentence[:-1]  # Remove the last character from the sentence
                print("Removed last character")

        # Check if the user presses 'q' to exit the loop
        if key == ord("q"):
            break

    except Exception as e:
        print(f"An error occurred: {e}")

# Print the final recognized sentence
print(f"Recognized sentence: {sentence}")


cap.release()
cv2.destroyAllWindows()

