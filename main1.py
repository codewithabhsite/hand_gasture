import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Draw styles
face_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
lip_spec = mp_drawing.DrawingSpec(color=(255,150,0), thickness=1, circle_radius=1)
eye_spec = mp_drawing.DrawingSpec(color=(0,200,255), thickness=1, circle_radius=1)
hand_spec = mp_drawing.DrawingSpec(color=(255,75,75), thickness=2, circle_radius=2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam not found or cannot be opened!")
    exit()

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
) as face_mesh, \
     mp_hands.Hands(
         static_image_mode=False,
         max_num_hands=2,
         min_detection_confidence=0.7,
         min_tracking_confidence=0.7,
     ) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face mesh detection
        face_results = face_mesh.process(rgb_frame)
        annotated_frame = frame.copy()

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=face_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=lip_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,150,0), thickness=2)
                )
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=eye_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,255), thickness=2)
                )
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=eye_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,255), thickness=2)
                )

        # Hand landmarks/gesture detection
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    hand_spec,
                    mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2)
                )
                # Hand gesture logic add कर सकते हो (जैसे pinch, fist, etc.)

        cv2.imshow('Face + Hand Landmarks Clean Display', annotated_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
