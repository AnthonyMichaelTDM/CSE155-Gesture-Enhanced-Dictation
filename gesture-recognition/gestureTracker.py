import cv2 as cv  # OpenCV library for computer vision tasks
import mediapipe as mp  # MediaPipe library for hand tracking and gesture recognition
import redis 
import threading  # Import threading for non-blocking sound
import pygame  # Import the pygame library for sound

# Initialize Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)  # Connect to local Redis instance

# Initialize MediaPipe's drawing utilities and styles for visualizing landmarks on the hand
mp_drawing = mp.solutions.drawing_utils  # Predefined module for drawing landmarks on images
mp_drawing_style = mp.solutions.drawing_styles  # Predefined styles for drawing hand landmarks
mphands = mp.solutions.hands  # Predefined module for hand detection and tracking

# Start video capture from the default camera (0 typically refers to the built-in webcam)
cap = cv.VideoCapture(0)

# Initialize the MediaPipe hands model, which will detect and track hand landmarks
hands = mphands.Hands()  # This loads the hand landmark model

print("Program started.")

# Variables to track previous gesture states so we don't keep continously pushing the puncuation into the queue since it's an infinite loop
prev_index_finger_extended = False
prev_middle_finger_extended = False

# Initialize pygame mixer
pygame.mixer.init()

def play_chime(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

chime_file = "Chime.mp3"  # Path to your sound file

# Infinite loop to process video frames in real-time
while True:
    # Capture the video frame from the camera
    data, image = cap.read()
    
    # Flip the image horizontally for natural viewing (like looking in a mirror)
    # OpenCV uses BGR color format by default, but MediaPipe expects RGB
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    
    # Process the image to detect hands and hand landmarks using MediaPipe's model
    results = hands.process(image)
    
    # Convert the image back from RGB to BGR so OpenCV can display it properly
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    # Check if any hands were detected in the current frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, 
                mphands.HAND_CONNECTIONS
            )

            # Get coordinates of relevant landmarks
            index_finger_tip = hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_base = hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_MCP]  # Base of index finger (MCP)
            wrist = hand_landmarks.landmark[mphands.HandLandmark.WRIST]  # Wrist landmark

            # Print debug information
            # print(f"Index Finger Tip: ({index_finger_tip.x}, {index_finger_tip.y}), Base: ({index_finger_base.x}, {index_finger_base.y}), Wrist: ({wrist.x}, {wrist.y})")

            # Check if the index finger is raised
            index_finger_extended = index_finger_tip.y < wrist.y and (index_finger_tip.y < index_finger_base.y - 0.05)

            # Check the positions of other fingers
            middle_finger_tip = hand_landmarks.landmark[mphands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mphands.HandLandmark.RING_FINGER_TIP]
            pinky_finger_tip = hand_landmarks.landmark[mphands.HandLandmark.PINKY_TIP]
            thumb_tip = hand_landmarks.landmark[mphands.HandLandmark.THUMB_TIP]

            # Check if the other fingers are closed (lower than their respective bases)
            # In most image processing libraries, including OpenCV, the y-axis increases as you go downward and decreases as you go upward
            middle_finger_closed = middle_finger_tip.y > hand_landmarks.landmark[mphands.HandLandmark.MIDDLE_FINGER_MCP].y
            ring_finger_closed = ring_finger_tip.y > hand_landmarks.landmark[mphands.HandLandmark.RING_FINGER_MCP].y
            pinky_finger_closed = pinky_finger_tip.y > hand_landmarks.landmark[mphands.HandLandmark.PINKY_MCP].y
            
            # Print debug information for finger statuses
            # print(f"Index Finger Extended: {index_finger_extended}, Middle Finger Closed: {middle_finger_closed}, Ring Finger Closed: {ring_finger_closed}, Pinky Finger Closed: {pinky_finger_closed}")

            """ Look into doing match cases for the if statements """

            # Check if the index finger is the only finger raised
            if index_finger_extended and middle_finger_closed and ring_finger_closed and pinky_finger_closed:
                # Print a message on the frame in blue color when the index finger is the only finger raised
                cv.putText(image, "Only Index Finger Raised!", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                if not prev_index_finger_extended:  # If the state has changed (finger was not raised before)
                    # Push a punctuation object (e.g., ".") to Redis queue when the gesture is detected
                    # The queue is named 'gesture_queue' here but it can be whatever
                    r.lpush('gesture_queue', '.')  # You can customize the data
                    # Play the sound on a separate thread to avoid blocking the program
                    threading.Thread(target=play_chime, args=(chime_file,)).start()
                    print("Pushed '.' to the queue!")  # Debug print to see in terminal
                    # Popping an item from the Redis queue
                    gesture = r.lpop('gesture_queue')
                    if gesture:
                        print(f"Popped gesture: {gesture.decode('utf-8')}")  # Decode bytes to string
                        print()
                    else:
                        print("No gestures in the queue.")

            prev_index_finger_extended = index_finger_extended  # Update previous state

            # Check if both index and middle fingers are raised
            middle_finger_extended = middle_finger_tip.y < wrist.y and (middle_finger_tip.y < hand_landmarks.landmark[mphands.HandLandmark.MIDDLE_FINGER_MCP].y - 0.05)
            if index_finger_extended and middle_finger_extended and ring_finger_closed and pinky_finger_closed:
                # Print a message on the frame in green color when both fingers are raised
                cv.putText(image, "Index & Middle Fingers Raised!", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                if not (prev_index_finger_extended and prev_middle_finger_extended):  # Check if both states changed
                    r.lpush('gesture_queue', ',')  # You can customize the data
                    # Play the sound on a separate thread to avoid blocking the program
                    threading.Thread(target=play_chime, args=(chime_file,)).start()
                    print("Pushed ',' to the queue!")  # Debug print to see in terminal
                    gesture = r.lpop('gesture_queue')
                    if gesture:
                        print(f"Popped gesture: {gesture.decode('utf-8')}")  # Decode bytes to string
                        print()
                    else:
                        print("No gestures in the queue.")
            prev_middle_finger_extended = middle_finger_extended  # Update previous state


    
    # Display the resulting frame with hand landmarks drawn on it
    cv.imshow("HandTracker", image)
    
    # Wait for a key press for 1 millisecond before showing the next frame
    # This also allows OpenCV to keep the window responsive
    cv.waitKey(1)


# add a feature that disables the sound