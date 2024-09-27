import cv2
from mtcnn import MTCNN
import numpy as np
import os
from gcn import refine_landmarks

video_directory = r"C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_ETHIOS\basic_vid_process\Videos\Videos\run"
output_directory = r"C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_ETHIOS\basic_vid_process\Videos\Videos\outputs"

os.makedirs(output_directory, exist_ok=True)
video_files = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]

if not video_files:
    print("No video files found in the specified directory!")
else:
    print(f"Video files found: {video_files}")

detector = MTCNN()
if video_files:
    video_file = video_files[0]  
    video_path = os.path.join(video_directory, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_file}")
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_path = os.path.join(output_directory, f"processed_{video_file}")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        print(f"Processing video: {video_file}, Output: {output_path}")

        frame_skip_interval = 5
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            frame_counter += 1

            if frame_counter % frame_skip_interval != 0:
                continue

            if not ret or frame is None:
                print(f"Finished processing video: {video_file}")
                break

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
                    cv2.circle(frame, center=(int(x_lm), int(y_lm)), color=(255, 0, 0), thickness=2, radius=5)

                refined_landmarks = refine_landmarks(landmarks)

                for (x_lm_refined, y_lm_refined) in refined_landmarks:
                    cv2.circle(frame, center=(int(x_lm_refined), int(y_lm_refined)), color=(0, 255, 0), thickness=2, radius=5)

            out.write(frame)
            cv2.imshow(f'Landmark Detection (Video: {video_file})', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        print(f"Processed video saved as: {output_path}")

cv2.destroyAllWindows()
print(f"Check the output folder for processed file at: {output_directory}")
