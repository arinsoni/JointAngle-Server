import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator(VideoTransformerBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        results = self.pose.process(img)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

def get_angle_vertical(joint1, joint2):
    j1 = np.array(joint1)
    j2 = np.array(joint2)
    vector = j1 - j2
    vertical_vector = np.array([0, 1])
    dot_product = np.dot(vector, vertical_vector)
    magnitude_vector = np.linalg.norm(vector)
    cosine_angle = dot_product / magnitude_vector
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def main():
    st.title("Live Pose Estimation using MediaPipe and Streamlit WebRTC")
    
    webrtc_ctx = webrtc_streamer(key="pose_estimation", 
                                 video_transformer_factory=PoseEstimator,
                                 media_stream_constraints={"video": True, "audio": False},
                                 async_transform=True)

    if webrtc_ctx.video_transformer:
        st.write("Video transformer is active. Now streaming and processing video from your browser!")

if __name__ == "__main__":
    main()
