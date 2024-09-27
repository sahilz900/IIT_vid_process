import cv2
from mtcnn import MTCNN
import numpy as np
from gcn import refine_landmarks

# Load the video file
video_path = r"C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_ETHIOS\basic_vid_process\50_data\vid3.mp4"
cap = cv2.VideoCapture(video_path)

# Define the codec and create a VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4
output_path = r"C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_ETHIOS\basic_vid_process\output\output_vid3.mp4"
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

detector = MTCNN()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame or end of video")
        break

    # Rotate the frame 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    output = detector.detect_faces(frame)

    for i in output:
        x, y, width, height = i['box']
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + width, y + height), color=(0, 0, 255), thickness=2)

        landmarks = np.array([
            [i['keypoints']['left_eye'][0], i['keypoints']['left_eye'][1]],
            [i['keypoints']['right_eye'][0], i['keypoints']['right_eye'][1]],
            [i['keypoints']['nose'][0], i['keypoints']['nose'][1]],
            [i['keypoints']['mouth_left'][0], i['keypoints']['mouth_left'][1]],
            [i['keypoints']['mouth_right'][0], i['keypoints']['mouth_right'][1]]
        ], dtype=np.float32)

        for (x_lm, y_lm) in landmarks:
            cv2.circle(frame, center=(int(x_lm), int(y_lm)), color=(255, 0, 0), thickness=2, radius=5)  # Blue for original

        refined_landmarks = refine_landmarks(landmarks)

        for (x_lm_refined, y_lm_refined) in refined_landmarks:
            cv2.circle(frame, center=(int(x_lm_refined), int(y_lm_refined)), color=(0, 255, 0), thickness=2, radius=5)  # Green for refined

    # Write the frame to the output video
    out.write(frame)

    cv2.imshow('Landmark Detection (Blue: Original, Green: Refined)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()
