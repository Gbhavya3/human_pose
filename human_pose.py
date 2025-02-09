import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents interactive mode
import matplotlib.pyplot as plt
import streamlit as st

# Define body parts
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5,
    "LElbow": 6, "LWrist": 7, "RHip": 8, "LHip": 9, "RKnee": 10, "LKnee": 11,
    "RAnkle": 12, "LAnkle": 13, "REye": 14, "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

# Define pose pairs
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["Neck", "RHip"], ["Neck", "LHip"],
    ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
    ["Neck", "Nose"], ["Nose", "REye"], ["Nose", "LEye"], ["REye", "REar"], ["LEye", "LEar"]
]

# Load model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
thres = 0.2

def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    net.getPerfProfile()
    return frame

# Read input image
input_img = cv2.imread('stand.jpg')
output_img = poseDetector(input_img)

# Save output image
cv2.imwrite("Output-Image.png", output_img)

# Streamlit Integration
st.subheader("Positions Estimated")
st.image(output_img, caption="Positions Estimated", use_column_width=True)

thres = st.slider("Threshold for detecting the key points", min_value=0, max_value=100, value=20) / 100

@st.cache_data
def poseDetectorStreamlit(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
    
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    
    net.getPerfProfile()
    return frame

output_streamlit = poseDetectorStreamlit(input_img)
st.image(output_streamlit, caption="Processed Pose Estimation", use_column_width=True)
