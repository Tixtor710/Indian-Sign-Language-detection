# Step 1: Train and Export Model (Kaggle Notebook)
# ==================================================

# Skipped this step as ONNX model already exists

# ==================================================
# Step 2: Real-Time Inference Script (Local OpenCV + ONNX Runtime)
# ==================================================

import cv2
import numpy as np
import os

# Use local path to ONNX model
onnx_model_path = os.path.join(os.getcwd(), "C:\ISL\isl_cnn_model .onnx")
if not os.path.exists(onnx_model_path):
    print("ONNX model not found. Please export your trained model to ONNX format.")
    exit(1)

net = cv2.dnn.readNetFromONNX(onnx_model_path)

classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
background = None
accumulated_weight = 0.5
ROI_top, ROI_bottom = 100, 300
ROI_right, ROI_left = 150, 350

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    contours, _ = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return frame, max(contours, key=cv2.contourArea)

cam = cv2.VideoCapture(0)
num_frames = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    if roi.shape[0] == 0 or roi.shape[1] == 0:
        continue

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 40, 30], dtype="uint8")
    upper = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(hsv, lower, upper)
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(gray, gray, mask=skinMask)
    processed = cv2.Canny(skin, 60, 60)

    if num_frames < 70:
        cal_accum_avg(processed, accumulated_weight)
        cv2.putText(frame_copy, "Calibrating background...", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        result = segment_hand(processed)
        if result:
            thresholded, hand_segment = result
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            thresholded = cv2.resize(thresholded, (128, 128))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            input_blob = cv2.dnn.blobFromImage(thresholded, scalefactor=1.0/255.0, size=(128, 128), mean=(0,0,0), swapRB=True)

            try:
                net.setInput(input_blob)
                preds = net.forward()
                label_idx = np.argmax(preds[0])
                cv2.putText(frame_copy, classes[label_idx], (240, 86), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                cv2.putText(frame_copy, f'Prediction Failed: {str(e)}', (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
    num_frames += 1
    cv2.putText(frame_copy, "Real-Time ISL Detection", (10, 20), cv2.FONT_ITALIC, 0.7, (255,255,255), 1)
    cv2.imshow("ISL Detection", frame_copy)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
del net
cv2.destroyAllWindows()
