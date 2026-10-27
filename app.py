
import os
import time
import cv2
import numpy as np
import mediapipe as mp
import joblib
from skimage.feature import hog

# =====================================================
# CONFIGURATION
# =====================================================
WIDTH, HEIGHT = 960, 540
CAMERA_INDEX = 0
MODEL_PATH = "models/drawing_model.joblib"

# Drawing settings
DRAW_THICKNESS = 7
ERASER_RADIUS = 30

# Pinch hysteresis prevents accidental start/stop flickering
PINCH_START = 0.055
PINCH_RELEASE = 0.085

# Adaptive smoothing: responsive for fast movements,
# smoother for slow movements
MIN_SMOOTHING = 0.28
MAX_SMOOTHING = 0.85
SPEED_FACTOR = 0.012

PREDICTION_INTERVAL = 0.7

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# =====================================================
# LOAD MODEL
# =====================================================
model = None
encoder = None

if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    encoder = bundle["encoder"]
    print("AI model loaded.")
else:
    print("AI model missing. Drawing works without predictions.")
    print("Train it with: python train_model.py")

# =====================================================
# HAND TRACKING
# =====================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.65,
)

# =====================================================
# CAMERA
# =====================================================
cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    hands.close()
    raise RuntimeError("Webcam unavailable. Try CAMERA_INDEX = 1.")

# Persistent layers
drawing_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
drawing_layer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

previous_point = None
smoothed_point = None
drawing_active = False
eraser_enabled = False

predictions = []
last_draw_time = 0.0
last_prediction_time = 0.0

print("""
========= SMOOTH AIR DRAWING AI =========
Pinch thumb + index : Draw
Release pinch       : Stop drawing
E                   : Toggle eraser
C                   : Clear drawing
S                   : Save camera image
Q                   : Quit
=========================================
""")


# =====================================================
# SMOOTHING
# =====================================================
def smooth_point(point):
    global smoothed_point

    if smoothed_point is None:
        smoothed_point = np.array(point, dtype=np.float32)
        return tuple(np.round(smoothed_point).astype(int))

    target = np.array(point, dtype=np.float32)
    movement = float(np.linalg.norm(target - smoothed_point))

    # Slow movement gets more smoothing; fast movement
    # gets a quicker response to reduce trailing.
    alpha = MIN_SMOOTHING + movement * SPEED_FACTOR
    alpha = float(np.clip(alpha, MIN_SMOOTHING, MAX_SMOOTHING))

    smoothed_point += alpha * (target - smoothed_point)

    return tuple(np.round(smoothed_point).astype(int))


# =====================================================
# DRAWING
# =====================================================
def add_stroke(start, end):
    """Write into both the AI mask and visible color layer."""
    cv2.line(
        drawing_mask, start, end,
        255, DRAW_THICKNESS, cv2.LINE_AA
    )

    cv2.line(
        drawing_layer, start, end,
        YELLOW, DRAW_THICKNESS, cv2.LINE_AA
    )


def erase_at(point):
    cv2.circle(
        drawing_mask, point, ERASER_RADIUS, 0, -1
    )
    cv2.circle(
        drawing_layer, point, ERASER_RADIUS, (0, 0, 0), -1
    )


def clear_drawing():
    global previous_point, smoothed_point
    global drawing_active, predictions

    drawing_mask[:] = 0
    drawing_layer[:] = 0

    previous_point = None
    smoothed_point = None
    drawing_active = False
    predictions = []


def overlay_strokes(frame):
    """Render persistent strokes on top of every camera frame."""
    visible = np.any(drawing_layer != 0, axis=2)
    frame[visible] = drawing_layer[visible]


# =====================================================
# AI RECOGNITION
# =====================================================
def extract_features(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) < 25:
        return None

    crop = mask[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    side = max(crop.shape) + 30
    square = np.zeros((side, side), dtype=np.uint8)

    x_offset = (side - crop.shape[1]) // 2
    y_offset = (side - crop.shape[0]) // 2

    square[
        y_offset:y_offset + crop.shape[0],
        x_offset:x_offset + crop.shape[1]
    ] = crop

    image = cv2.resize(square, (64, 64))
    image = 255 - image

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    return features.reshape(1, -1).astype(np.float32)


def predict_object():
    if model is None:
        return []

    features = extract_features(drawing_mask)

    if features is None:
        return []

    scores = np.asarray(model.decision_function(features))

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    scores = scores[0]
    indices = np.argsort(scores)[::-1][:3]
    best_scores = scores[indices]

    # Relative ranking scores, not true probabilities
    weights = np.exp(np.clip(best_scores - best_scores.max(), -30, 0))
    weights /= weights.sum() + 1e-9

    return [
        (
            encoder.inverse_transform([int(idx)])[0],
            float(score)
        )
        for idx, score in zip(indices, weights)
    ]


# =====================================================
# INTERFACE
# =====================================================
def draw_interface(frame, mode, cursor_found):
    panel = frame.copy()
    cv2.rectangle(panel, (8, 8), (390, 150), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)

    cv2.putText(
        frame, "SMOOTH AIR DRAWING AI", (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2
    )

    color = (
        YELLOW if mode == "DRAWING"
        else RED if mode == "ERASING"
        else GREEN if cursor_found
        else RED
    )

    cv2.putText(
        frame, f"MODE: {mode}", (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2
    )

    if model is None:
        cv2.putText(
            frame, "AI model not loaded", (18, 89),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1
        )
    elif predictions:
        for i, (label, score) in enumerate(predictions):
            cv2.putText(
                frame, f"{i + 1}. {label} ({score:.0%} relative)",
                (18, 88 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                GREEN if i == 0 else WHITE, 1
            )

    cv2.putText(
        frame, "Pinch: Draw | E: Erase | C: Clear | S: Save | Q: Quit",
        (10, HEIGHT - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, WHITE, 1
    )


# =====================================================
# MAIN LOOP
# =====================================================
try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Camera frame unavailable.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        mode = "NO HAND"
        cursor = None

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            # Index fingertip controls the cursor
            raw_point = (
                int(lm[8].x * WIDTH),
                int(lm[8].y * HEIGHT),
            )

            raw_point = (
                np.clip(raw_point[0], 0, WIDTH - 1).item(),
                np.clip(raw_point[1], 0, HEIGHT - 1).item(),
            )

            cursor = smooth_point(raw_point)

            # Pinch distance is independent of screen resolution
            pinch_distance = float(np.hypot(
                lm[4].x - lm[8].x,
                lm[4].y - lm[8].y,
            ))

            # Hysteresis: use different thresholds to start and
            # stop drawing, avoiding rapid toggling.
            if not drawing_active and pinch_distance < PINCH_START:
                drawing_active = True
            elif drawing_active and pinch_distance > PINCH_RELEASE:
                drawing_active = False
                previous_point = None

            if eraser_enabled:
                mode = "ERASING"
                drawing_active = False
                previous_point = None

                cv2.circle(
                    frame, cursor, ERASER_RADIUS, RED, 2
                )
                erase_at(cursor)
                predictions = []

            elif drawing_active:
                mode = "DRAWING"

                if previous_point is None:
                    cv2.circle(
                        drawing_mask, cursor,
                        DRAW_THICKNESS // 2, 255, -1
                    )
                    cv2.circle(
                        drawing_layer, cursor,
                        DRAW_THICKNESS // 2, YELLOW, -1
                    )
                else:
                    # Anti-aliased line interpolation fills gaps
                    # between successive fingertip positions.
                    add_stroke(previous_point, cursor)

                previous_point = cursor
                last_draw_time = time.time()

            else:
                mode = "READY - PINCH TO DRAW"
                previous_point = None

            # Clear cursor indicator
            cv2.circle(
                frame, cursor, 8,
                GREEN if not drawing_active else YELLOW, 2
            )

        else:
            previous_point = None
            smoothed_point = None
            drawing_active = False

        now = time.time()

        # Recognize after a short pause to avoid excessive CPU use.
        if (
            model is not None
            and np.any(drawing_mask)
            and now - last_draw_time > 0.4
            and now - last_prediction_time > PREDICTION_INTERVAL
        ):
            try:
                predictions = predict_object()
            except Exception as error:
                print("AI prediction error:", error)

            last_prediction_time = now

        # Always show previous strokes over the live camera.
        overlay_strokes(frame)
        draw_interface(frame, mode, cursor is not None)

        cv2.imshow("Smooth Air Drawing AI", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            clear_drawing()

        elif key == ord("e"):
            eraser_enabled = not eraser_enabled
            drawing_active = False
            previous_point = None

        elif key == ord("s"):
            filename = f"air_drawing_{int(time.time())}.png"
            if cv2.imwrite(filename, frame):
                print("Saved:", filename)

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
