# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# import base64
# from io import BytesIO
# import mediapipe as mp

# application = Flask(__name__)

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# def get_angle(joint1, joint2, joint3):
#     j1 = np.array(joint1)
#     j2 = np.array(joint2) 
#     j3 = np.array(joint3) 
    
#     radians = np.arctan2(
#         j3[1] - j2[1], j3[0] - j2[0]) - np.arctan2(j1[1] - j2[1], j1[0] - j2[0])

#     angle = np.abs(radians * 180.0 / np.pi)
    
#     if angle > 180.0:
#         angle = 360 - angle
        
#     return angle 

# pose_dict = mp.solutions.pose.PoseLandmark

# body_parts = [
#     {
#         "name": "left_arm",
#         "components": [
#             mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
#             mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
#             mp.solutions.pose.PoseLandmark.LEFT_WRIST
#         ]
#     },
#     {
#         "name": "right_arm",
#         "components": [
#             mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
#             mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
#             mp.solutions.pose.PoseLandmark.RIGHT_WRIST
#         ]
#     },
#     {
#         "name": "right_leg",
#         "components": [
#             mp.solutions.pose.PoseLandmark.RIGHT_HIP,
#             mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
#             mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
#         ]
#     },
#     {
#         "name": "left_leg",
#         "components": [
#             mp.solutions.pose.PoseLandmark.LEFT_HIP,
#             mp.solutions.pose.PoseLandmark.LEFT_KNEE,
#             mp.solutions.pose.PoseLandmark.LEFT_ANKLE
#         ]
#     }
# ]



# print(body_parts[0]["components"][0])
# @application.route('/process_image', methods=['POST'])
# def process_image():
#     data = request.get_json()
#     if not data or 'image' not in data:
#         return jsonify({'error': 'No image data provided'}), 400

#     base64_image = data['image']
#     image_data = base64.b64decode(base64_image)
#     nparr = np.frombuffer(image_data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     results = pose.process(img)
#     try:
#         landmarks = results.pose_landmarks.landmark
#         angles_response = [{"name": part["name"], "angle": []} for part in body_parts]
            
#         for part, response_part in zip(body_parts, angles_response):
#             joint1_idx, joint2_idx, joint3_idx = [component.value for component in part["components"]]
#             joint1 = (landmarks[joint1_idx].x, landmarks[joint1_idx].y)
#             joint2 = (landmarks[joint2_idx].x, landmarks[joint2_idx].y)
#             joint3 = (landmarks[joint3_idx].x, landmarks[joint3_idx].y)
#             angle = get_angle(joint1, joint2, joint3)
#             response_part["angle"].append(angle)

#         print(angles_response)
#         return jsonify(angles_response)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     application.run(debug=True)


from flask import Flask
application = Flask(__name__)

@application.route("/")
def hello_world():
    return "<h1 style='color:green'>Hello World!</h1>"

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000)