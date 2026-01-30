import cv2
import mediapipe as mp
import json
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

def get_limb_unit_vector(p1, p2):
    """Calculates the unit vector between two landmarks to ensure scale invariance."""
    vec = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else [0, 0, 0]

def extract_ghost(video_path, output_json):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) [cite: 200]
    ghost_data = {
        "metadata": {"source": video_path, "target_fps": 30},
        "frames": []
    }

    print(f"Processing: {video_path} at {fps} FPS...")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Process every Nth frame to normalize to 30 FPS if source is higher
        if frame_count % (max(1, int(fps / 30))) != 0:
            frame_count += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame) [cite: 176]

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Root-Relative Centering: Origin (0,0,0) is Mid-Hip [cite: 75, 76, 77]
            mid_hip_x = (lm[23].x + lm[24].x) / 2
            mid_hip_y = (lm[23].y + lm[24].y) / 2
            mid_hip_z = (lm[23].z + lm[24].z) / 2

            # Extract orientation of critical limbs (Limb Vector Extraction) [cite: 79, 80]
            # 1. Right Humerus (Shoulder to Elbow) [cite: 86]
            r_humerus = get_limb_unit_vector(lm[12], lm[14])
            # 2. Right Radius (Elbow to Wrist) [cite: 87]
            r_radius = get_limb_unit_vector(lm[14], lm[16])
            
            ghost_data["frames"].append({
                "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
                "pose_embedding": r_humerus + r_radius # Concatenated 6D vector [cite: 88, 89]
            })

        frame_count += 1

    with open(output_json, 'w') as f:
        json.dump(ghost_data, f, indent=4)
    
    cap.release()
    print(f"Extraction complete. Ghost saved to {output_json}")

# Usage
# extract_ghost("master_palm_strike.mp4", "palm_strike_ghost.json")