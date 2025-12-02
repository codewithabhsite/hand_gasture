import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Neon Tech Color Palette ---
NEON_BLUE = (255, 120, 0)
NEON_PURPLE = (255, 0, 180)
NEON_CYAN = (255, 255, 0)
NEON_PINK = (180, 0, 255)
WHITE = (255, 255, 255)
ACCENT_GREEN = (100, 255, 150)

# --- Modern Hand UI Class ---
class ModernUI:
    def __init__(self):
        self.frame_count = 0
        self.gesture_state = "IDLE"

    def draw_status_panel(self, img, gesture, fps, expression):
        """Draw modern status panel with hand/facial info."""
        panel_height, panel_width = 100, 340
        x, y = 10, 10
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (x, y), (x + panel_width, y + panel_height), NEON_BLUE, 2)
        cv2.putText(img, 'AR Face + Hand UI', (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        cv2.putText(img, f'Hand: {gesture}', (x + 20, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_CYAN, 2)
        cv2.putText(img, f'Face: {expression}', (x + 20, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_GREEN, 2)
        cv2.putText(img, f'FPS: {fps}', (x + 240, y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEON_PINK, 2)
        time_str = datetime.now().strftime('%H:%M:%S')
        cv2.putText(img, time_str, (x + 240, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    def draw_open_hand_ui(self, img, palm, fingertips, angle):
        for idx, tip in enumerate(fingertips):
            color = NEON_CYAN if idx % 2 == 0 else NEON_PURPLE
            cv2.line(img, palm, tip, color, 2)
            cv2.circle(img, tip, 10, color, -1)
        # Core glow
        cv2.circle(img, palm, 30, NEON_PURPLE, -1)
        cv2.circle(img, palm, 32, NEON_PINK, 2)
        cv2.putText(img, f'Open Hand ({angle}°)', (palm[0] + 10, palm[1] - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, NEON_BLUE, 2)

    def draw_pinch_ui(self, img, palm, pinch_value):
        pulse = int(15 + 6 * np.sin(self.frame_count * 0.2))
        cv2.circle(img, palm, 35 + pulse, NEON_PINK, 2)
        cv2.circle(img, palm, 24, NEON_PURPLE, -1)
        cv2.ellipse(img, palm, (45, 45), 0, 0, int(pinch_value * 3.6), ACCENT_GREEN, 4)
        cv2.putText(img, f'Pinch {pinch_value}%', (palm[0] - 25, palm[1] + 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, NEON_CYAN, 2)

    def draw_fist_ui(self, img, palm):
        cv2.circle(img, palm, 20, NEON_CYAN, -1)
        cv2.circle(img, palm, 25, NEON_BLUE, 2)
        cv2.putText(img, 'FIST Locked', (palm[0] - 45, palm[1] - 30), cv2.FONT_HERSHEY_DUPLEX, 1, NEON_BLUE, 2)

# --- Facial Expression Enhancer ---
def get_face_expression(landmarks):
    # Use normalized mouth/eye/brow ratios for expression (blendshape alternative)
    mouth_left = landmarks.landmark[61]
    mouth_right = landmarks.landmark[291]
    mouth_top = landmarks.landmark[13]
    mouth_bottom = landmarks.landmark[14]
    brow_left = landmarks.landmark[70]
    brow_right = landmarks.landmark[300]
    eye_left = landmarks.landmark[159]
    eye_right = landmarks.landmark[386]
    mouth_width = np.linalg.norm([mouth_right.x - mouth_left.x, mouth_right.y - mouth_left.y])
    mouth_height = np.linalg.norm([mouth_top.x - mouth_bottom.x, mouth_top.y - mouth_bottom.y])
    brow_raise_left = eye_left.y - brow_left.y
    brow_raise_right = eye_right.y - brow_right.y
    avg_brow_raise = (brow_raise_left + brow_raise_right) / 2
    if mouth_height / mouth_width > 0.43:
        return "Frown 😞"
    elif mouth_width > 0.41 and mouth_height / mouth_width < 0.22:
        return "Happy 😊"
    elif avg_brow_raise > 0.024 and mouth_height / mouth_width < 0.3:
        return "Surprised 😮"
    else:
        return "Neutral 😐"

# --- Main UI/Detection Loop ---
ui = ModernUI()
cap = cv2.VideoCapture(0)
prev_time = 0

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as face_mesh, mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Face Mesh + Enhanced Expression ---
        face_results = face_mesh.process(rgb_frame)
        expression_label = ""
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(60,190,255), thickness=2))
                expression_label = get_face_expression(face_landmarks)

        # --- Hand Gesture UI Detection ---
        detected_gesture_state = "NO HAND"
        if hands:
            hands_results = hands.process(rgb_frame)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    lm = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]
                    palm = lm[9]
                    fingertips = [lm[i] for i in [4, 8, 12, 16, 20]]
                    dists = [np.linalg.norm(np.array(tip) - np.array(palm)) for tip in fingertips]
                    avg_dist = np.mean(dists)
                    pinch_dist = np.linalg.norm(np.array(lm[4]) - np.array(lm[8]))
                    pinch_value = int(100 - min(pinch_dist, 100))
                    # Open Hand
                    if avg_dist > 70:
                        detected_gesture_state = "OPEN HAND"
                        v1 = np.array(lm[4]) - np.array(palm)
                        v2 = np.array(lm[8]) - np.array(palm)
                        try:
                            angle = int(np.degrees(np.arccos(
                                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                            )))
                        except:
                            angle = 0
                        ui.draw_open_hand_ui(frame, palm, fingertips, angle)
                    # Pinch
                    elif pinch_value > 45:
                        detected_gesture_state = "PINCH"
                        ui.draw_pinch_ui(frame, palm, pinch_value)
                    # Fist
                    else:
                        detected_gesture_state = "FIST"
                        ui.draw_fist_ui(frame, palm)
            else:
                detected_gesture_state = "NO HAND"
        # --- Status Panel ---
        current_time = cv2.getTickCount()
        fps = int(cv2.getTickFrequency() / (current_time - prev_time)) if prev_time > 0 else 0
        prev_time = current_time
        ui.draw_status_panel(frame, detected_gesture_state, fps, expression_label)

        ui.frame_count += 1
        cv2.imshow('Modern Face+Hand AR UI', frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
