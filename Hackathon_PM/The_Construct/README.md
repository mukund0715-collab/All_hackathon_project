🟢 Project Morpheus: The Construct
An Offline, Privacy-First AI Self-Defense Trainer for Women.
🚨 The Problem

Traditional AI fitness or self-defense apps require users to upload videos of themselves to cloud servers for processing. For a women's safety app, this is a massive privacy risk. Women should not have to compromise their digital privacy or record themselves in their homes just to learn basic self-defense. Furthermore, cloud-based LLMs introduce latency that ruins real-time physical training.

💡 The Solution

Project Morpheus is a 100% local, zero-cloud biometric training terminal.
It uses advanced 3D spatial tracking to compare a user's live movements against a pre-recorded "Ghost" of a martial arts master.

Instead of recording the user, the app acts as a mathematical mirror. It extracts skeletal data locally in real-time, calculates the vector differences (Dynamic Time Warping & Cosine Similarity), and immediately deletes the frame.

No video is ever saved. No data ever leaves the laptop.
✨ Key Features

    Zero-Cloud Architecture: All neural network processing (MediaPipe) happens directly on the user's local hardware.

    Deterministic Math-Based AI Coach: Instead of hallucination-prone, high-latency LLMs, Morpheus uses raw 3D kinematics to deliver instant, sub-millisecond coaching (e.g., "Your right elbow is misaligned during the extension phase").

    Semantic Action Tagging: The engine doesn't just look at raw numbers; it understands the phase of the movement (e.g., "Raising the knee", "Extending the hand") to provide context-aware feedback.

    The "Matrix" UI: A sleek, gamified CustomTkinter interface that separates the heavy OpenCV rendering from the user dashboard for maximum stability.

⚙️ How It Works (The Pipeline)

    The Master Extraction (Offline): We feed videos of self-defense experts into our engine. It extracts their perfect skeletal movements and saves them as lightweight .json vector files (The "Ghost").

    The Terminal UI (app.py): The user logs into a secure local session and selects a training module (e.g., Module N0: Palm Strike).

    The Comparator Engine (live4.py): The user performs the move in front of the webcam. The system overlays the Matrix HUD, tracking 33 body landmarks at 30 FPS.

    The After-Action Report (ai_coach1.py): The system analyzes the stuck_coordinates_log.json to tell the user exactly which joint failed at what specific phase of the move, granting a final sync score.

🛠️ Tech Stack

    Python 3.10+ (Core Engine)

    OpenCV (cv2) (Real-time video rendering and visual HUD)

    Google MediaPipe (Sub-millisecond pose estimation / computer vision)

    NumPy (Kinematic vector math, Cosine Similarity, Euclidean geometry)

    CustomTkinter (Modern, dark-theme GUI)

🎮 How to Use (Demo Flow)

Launch the app.py UI.

Enter an "Operator Alias" (Notice: No cloud password required, proving zero cloud auth).

Select Module N0: Palm Strike.

Click "View Master Tape" to see the reference movement.

Click "Enter The Construct" to activate the webcam tracker. Perform the move.

When the session ends, the terminal will launch the AI Coach, triggering audio feedback and exact geometric corrections.
