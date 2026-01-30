import cv2
import mediapipe as mp
import json
import time
import backend as be  # Imports our shim
import numpy as np    # Standard numpy for file I/O (JSON doesn't like CuPy)

# --- CONFIG ---
MOVE_NAME = "Palm Strike"
OUTPUT_FILE = "move_palm_strike.json"

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def main():
    cap = cv2.VideoCapture(0)
    recording = False
    frames_buffer = []
    
    print(f"--- GHOST RECORDER: {MOVE_NAME} ---")
    print("Press 'r' to START Recording")
    print("Press 's' to STOP and SAVE")
    print("Press 'q' to QUIT")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Recolor for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # --- RECORDING LOGIC ---
            if recording:
                # 1. Extract Raw Coordinates
                # We save only x, y, z, visibility
                raw_landmarks = []
                for lm in results.pose_landmarks.landmark:
                    raw_landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                
                # 2. Convert to hardware-agnostic array
                lm_array = be.np.array(raw_landmarks) 
                
                # 3. Normalize (Spatial Normalization) [cite: 76]
                norm_array = be.normalize_to_hip(lm_array[:, :3]) # Pass only x,y,z
                
                # 4. Append to Buffer (Must convert back to standard list for JSON)
                frames_buffer.append(norm_array.tolist())
                
                cv2.putText(image, f"REC: {len(frames_buffer)} frames", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # UI Overlay
        if recording:
            cv2.circle(image, (30, 30), 10, (0, 0, 255), -1) # Red Dot
        
        cv2.imshow('Ghost Recorder', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('r'):
            recording = True
            frames_buffer = []
            print("[STATUS] Recording Started...")
        elif key == ord('s'):
            recording = False
            print(f"[STATUS] Recording Stopped. Captured {len(frames_buffer)} frames.")
            save_move(frames_buffer)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def save_move(data):
    # JSON Schema Structure [cite: 192]
    move_data = {
        "meta": {
            "name": MOVE_NAME,
            "fps": 30,
            "total_frames": len(data),
            "date": time.strftime("%Y-%m-%d")
        },
        "landmarks_normalized": data # This is the "Ghost Data"
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(move_data, f)
    print(f"[SUCCESS] Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()