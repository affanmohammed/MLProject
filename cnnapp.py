import os
import cv2
import pickle
import pyttsx3 
import numpy as np
import streamlit as st
import mediapipe as mp
from pathlib import Path
from matplotlib import pyplot as plt
t2s = pyttsx3.init()

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

previous_prediction = None
sentence = []

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {1:'Hello',2:'Thumps Up',3:'I Love You',4:'Yes',5:'No',6:'Good Bye',7:"Cute",8:'Father',9:'Mother',10:'Why'}


cap = cv2.VideoCapture(0)
st.title("ASL with CNN")
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    #H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * 640) - 10
        y1 = int(min(y_) * 480) - 10

        x2 = int(max(x_) * 640) - 10
        y2 = int(max(y_) * 480) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        
        if len(sentence)>=0:
            if predicted_character != previous_prediction:
                sentence.append(predicted_character)
                t2s.say(predicted_character)
                t2s.runAndWait()
                previous_prediction = predicted_character
        
            if len(sentence) > 5:
                sentence = sentence[-5:]    

        cv2.rectangle(frame, (0, 0), (640, 40), (200, 100, 50), -1)
        cv2.putText(frame, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        #frame_placeholder.image(frame, channels="BGR")
   
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if stop_button_pressed:
        break


cap.release()
cv2.destroyAllWindows()
