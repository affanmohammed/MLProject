import cv2
import streamlit as st

run = st.checkbox('Run')
if run:
    cap = cv2.VideoCapture(1)
    st.title("test2 ")
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")

    while cap.isOpened():
        ret, frame = cap.read()
        cv2.rectangle(frame, (0, 0), (640, 40), (115,115,115), -1)
        frame_placeholder.image(frame, channels="BGR")

        if stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()
