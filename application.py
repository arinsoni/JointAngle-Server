import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_angle_vertical(joint1, joint2):
    j1 = np.array(joint1)
    j2 = np.array(joint2)
    vector = j1 - j2
    vertical_vector = np.array([0, 1])
    dot_product = np.dot(vector, vertical_vector)
    magnitude_vector = np.linalg.norm(vector)
    cosine_angle = dot_product / (magnitude_vector)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def find_camera_index():
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return index
    return None

def main():
    st.title("Live Pose Estimation")
    st.sidebar.header("Controls")
    
    camera_index = find_camera_index()
    if camera_index is None:
        st.error("No camera found. Please connect a camera and refresh the page.")
        return

    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

    start_camera = st.sidebar.button('Start Camera', key='start_camera')
    stop_camera = st.sidebar.button('Stop Camera', key='stop_camera')
    capture_button = st.sidebar.button('Capture', key='capture_frame')

    if start_camera:
        st.session_state.camera_on = True
    if stop_camera:
        st.session_state.camera_on = False

    FRAME_WINDOW = st.empty()
    cap = cv2.VideoCapture(camera_index)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while st.session_state.camera_on and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from camera.")
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            FRAME_WINDOW.image(image)

            if capture_button:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    angle = get_angle_vertical(shoulder, elbow)
                    st.write(f'Angle between shoulder and vertical: {angle:.2f} degrees')
                st.session_state.camera_on = False

    cap.release()

if __name__ == "__main__":
    main()
