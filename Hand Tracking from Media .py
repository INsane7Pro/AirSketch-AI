
import cv2
import mediapipe as mp
import time
import os
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------- CONFIGURATION --------------------

VIDEO_PATH = "a.mp4"
OUTPUT_PATH = "output_video.mp4"
MODEL_PATH = "hand_landmarker.task"

MAX_HANDS = 2
SHOW_LANDMARK_IDS = False

# Official MediaPipe hand landmark model
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# -------------------- MODEL SETUP --------------------

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Hand Landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video '{VIDEO_PATH}' not found. "
        "Place it in the project folder."
    )

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- VIDEO SETUP --------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH, fourcc, video_fps, (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

# -------------------- TRACKING LOOP --------------------

prev_time = time.perf_counter()
frame_index = 0
frame_interval_ms = 1000.0 / video_fps

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, img = cap.read()

        if not success:
            break

        frame_index += 1
        h, w = img.shape[:2]

        # MediaPipe Tasks requires an RGB image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        # Timestamps must increase for VIDEO mode
        timestamp_ms = int((frame_index - 1) * frame_interval_ms)
        result = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        detected_hands = len(result.hand_landmarks)

        # Draw each detected hand
        for hand_index, landmarks in enumerate(result.hand_landmarks):
            points = []

            for landmark in landmarks:
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                points.append((cx, cy))

            # Draw palm and finger connections
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    img, points[start], points[end],
                    (0, 220, 100), 2, cv2.LINE_AA
                )

            # Draw landmark points
            for landmark_id, (cx, cy) in enumerate(points):
                color = (255, 100, 0)

                # Fingertips: thumb, index, middle, ring, little
                if landmark_id in [4, 8, 12, 16, 20]:
                    color = (0, 165, 255)

                cv2.circle(
                    img, (cx, cy), 5, color, cv2.FILLED,
                    cv2.LINE_AA
                )

                if SHOW_LANDMARK_IDS:
                    cv2.putText(
                        img, str(landmark_id),
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1,
                        cv2.LINE_AA
                    )

            # Display handedness when available
            label = f"Hand {hand_index + 1}"

            if (
                hand_index < len(result.handedness)
                and result.handedness[hand_index]
            ):
                label = result.handedness[hand_index][0].category_name

            wrist_x, wrist_y = points[0]
            cv2.putText(
                img, label,
                (wrist_x, max(25, wrist_y - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2,
                cv2.LINE_AA
            )

        # Calculate display FPS safely
        current_time = time.perf_counter()
        elapsed = current_time - prev_time
        display_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        prev_time = current_time

        # Information panel
        cv2.rectangle(img, (10, 10), (290, 105), (30, 30, 30), -1)
        cv2.putText(
            img, f"FPS: {display_fps:.1f}",
            (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
        cv2.putText(
            img, f"Hands detected: {detected_hands}",
            (20, 68), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )
        cv2.putText(
            img, "Q: Quit | L: Landmark IDs",
            (20, 93), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (200, 200, 200), 1
        )

        cv2.imshow("Upgraded Hand Tracking", img)
        out.write(img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("l"):
            SHOW_LANDMARK_IDS = not SHOW_LANDMARK_IDS

# -------------------- CLEANUP --------------------

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete! Output saved to: {OUTPUT_PATH}")
