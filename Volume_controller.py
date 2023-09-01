import cv2
import numpy as np
import mediapipe as mp
import subprocess

mp_hands = mp.solutions.hands

# Initialize the hand tracking module
with mp_hands.Hands(max_num_hands=1) as hands:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Assume one hand is in the frame
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Get the coordinates of the thumb and index finger tips
            thumb_x, thumb_y = int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0])
            index_x, index_y = int(index.x * frame.shape[1]), int(index.y * frame.shape[0])

            # Calculate the Euclidean distance between thumb and index finger tips
            distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            # Map the distance to a volume range (adjust these values as needed)
            volume = int(np.interp(distance, [10, 250], [0, 100]))

            # Set the system volume on macOS using AppleScript and osascript
            volume_script = f'set volume output volume {volume}'
            subprocess.call(["osascript", "-e", volume_script])

            # Display volume level on the frame
            cv2.putText(frame, f"Volume: {volume}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand Volume Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
