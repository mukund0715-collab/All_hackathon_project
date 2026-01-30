import cv2
import mediapipe as mp
import json
import numpy as np

class GhostExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2) # Complexity 2 is "Heavy" for accuracy

    def get_unit_vector(self, p1, p2):
        v = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        mag = np.linalg.norm(v)
        return (v / mag).tolist() if mag > 1e-6 else [0, 0, 0]

    def extract(self, video_path, output_name):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        data = {"metadata": {"fps": 30, "total_frames": 0}, "frames": []}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Sub-sampling to 30 FPS [cite: 200, 201]
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % max(1, int(fps/30)) != 0: continue

            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_world_landmarks: # Use world_landmarks for real-world meter scale 
                lm = results.pose_world_landmarks.landmark
                
                # Full Biomechanical Chain 
                pose_vector = (
                    self.get_unit_vector(lm[11], lm[12]) + # Shoulders
                    self.get_unit_vector(lm[12], lm[14]) + # R-Upper-Arm [cite: 86]
                    self.get_unit_vector(lm[14], lm[16]) + # R-Forearm [cite: 87]
                    self.get_unit_vector(lm[11], lm[13]) + # L-Upper-Arm [cite: 84]
                    self.get_unit_vector(lm[13], lm[15]) + # L-Forearm [cite: 85]
                    self.get_unit_vector(lm[24], lm[26]) + # R-Thigh
                    self.get_unit_vector(lm[26], lm[28])   # R-Shin
                )
                data["frames"].append(pose_vector)

        data["metadata"]["total_frames"] = len(data["frames"])
        with open(f"{output_name}.json", 'w') as f:
            json.dump(data, f)
        cap.release()
