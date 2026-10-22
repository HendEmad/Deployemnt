import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import tempfile

st.title("Obejct Tracking App")
st.write("Upload a video to see object tracking in action.....")

def convert_color(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

uploaded_file = st.file_uploader("Choose an video...", type=["mp4", "avi", "mov"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    capture = cv2.VideoCapture(temp_file.name)   
    if not capture.isOpened():
        st.error("Error opening video file...")
    else:
        st.success("Video file opened successfully, processing starting ......")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Original video")
            stframe1 = st.empty()
        with col2:
            st.header("Processed video")
            stframe2 = st.empty()

        background_subtractor = cv2.createBackgroundSubtractorMOG2()
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        progress_bar = st.progress(0)
        frame_idx = 0

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            fg_mask = background_subtractor.apply(frame)
            (controus, _) = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in controus:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            stframe1.image(convert_color(frame), channels="RGB")
            stframe2.image(convert_color(fg_mask), channels="GRAY")
            frame_idx += 1
            progress_bar.progress(frame_idx / frame_count)


        capture.release()
        st.success("Video procesisng done...")