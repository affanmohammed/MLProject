import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing cv2

import cv2
import streamlit as st

def main():
    st.title("OpenCV + Streamlit Live Video Stream Example")

    run = st.checkbox('Run')

    if run:
        video_capture = cv2.VideoCapture(0)

        while run:
            ret, frame = video_capture.read()

            if not ret:
                st.error("Failed to capture frame.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, channels="RGB", use_column_width=True)

            # Wait for a short time to prevent freezing of the GUI
            key = cv2.waitKey(30) & 0xff
            if key == 27:  # Escape key to exit
                break

        video_capture.release()

if __name__ == "__main__":
    main()
