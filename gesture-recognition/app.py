#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2
import numpy as np
from mediapipe.python.solutions import hands as mp_hands

from utils.cvfpscalc import CvFpsCalc
from utils.draw import (
    draw_landmarks,
    draw_info_text,
    draw_info,
    draw_bounding_rect,
)
from model import KeyPointClassifier
from model import PointHistoryClassifier

from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
import time
import pygame
import threading


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=int,
        default=0.5,
    )

    args = parser.parse_args()

    return args


def play_chime(sound_file, muted):
    if not muted:
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    # with open(
    #     "model/point_history_classifier/point_history_classifier_label.csv",
    #     encoding="utf-8-sig",
    # ) as f:
    #     point_history_classifier_labels = csv.reader(f)
    #     point_history_classifier_labels = [
    #         row[0] for row in point_history_classifier_labels
    #     ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    # history_length = 16
    # point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    # finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    #  define redis queue #####################################################
    redis_connection = Redis(
        host="redis",
        # host="127.0.0.1",
        port=6379,
        retry_on_timeout=True,
        retry=Retry(backoff=ExponentialBackoff(), retries=3),
    )
    if redis_connection.ping():
        print("Connected to Redis successfully!")
        print()

    # Define variables for tracking gestures ##################################
    gesture_detected = False
    gesture_start_time = time.time()
    dwell_time = 3  # 3 seconds

    # define sound variables for successful queues and muted icon ##############
    pygame.mixer.init()
    muted = False
    chime_file = "Chime.mp3"

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == 109:  # m keybind (109 is m in ascii)
            muted = not muted
            if muted:
                print("Muted")
                print()
            else:
                print("Unmuted")
                print()
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks:

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # pre_processed_point_history_list = pre_process_point_history(
                #     debug_image, point_history
                # )
                # Write to the dataset file
                logging_csv(
                    number,
                    mode,
                    pre_processed_landmark_list,
                    # pre_processed_point_history_list,
                )

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Check if the detected gesture matches the desired punctuation ###################################
                punc_val = [".", ",", "?", "!", '"']

                # print(f"{keypoint_classifier_labels[hand_sign_id]}: {hand_sign_id}")
                if keypoint_classifier_labels[hand_sign_id] != "Neutral":
                    val_to_push = punc_val[hand_sign_id]
                    if not gesture_detected:
                        # Start timing the gesture
                        gesture_detected = True
                        gesture_start_time = time.time()
                        # makes the rectangle orange as the dwell time is processing
                        debug_image = draw_bounding_rect(
                            use_brect, debug_image, brect, "dwell"
                        )
                    elif time.time() - gesture_start_time >= dwell_time:
                        # Dwell time met, push to queue only once
                        redis_connection.rpush("gesture_queue", val_to_push)
                        # makes the rectangle green when the val is queued successfully
                        debug_image = draw_bounding_rect(
                            use_brect, debug_image, brect, "success"
                        )
                        threading.Thread(
                            target=play_chime, args=(chime_file, muted)
                        ).start()
                        print(f"Pushed {val_to_push} to queue.")

                        queued_val = redis_connection.lpop("gesture_queue")
                        if queued_val:
                            print(f"Dequeued {queued_val.decode()} from the queue.")
                            print()
                            # reset the reactangle  color back to normal
                            debug_image = draw_bounding_rect(
                                use_brect, debug_image, brect
                            )
                        gesture_detected = False  # Reset to prevent continuous queuing
                else:
                    # Reset if gesture is not detected in the frame
                    gesture_detected = False

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                )

        debug_image = draw_info(debug_image, fps, mode, number)

        # Display the mute icon if the sound is muted #####################################
        mute_icon = cv2.imread("volume-mute.png", cv2.IMREAD_UNCHANGED)
        if muted:  # Display the mute icon if the sound is muted
            if mute_icon is not None:  # Ensure the mute icon is loaded
                n = 70  # pixels
                mute_icon_resized = cv2.resize(
                    mute_icon, (n, n)
                )  # Resize the mute icon to nxn pixels

                h, w, _ = debug_image.shape  # Get the dimensions of the current frame
                x_offset, y_offset = w - (n + 10), h - (n + 10)  # Bottom-right corner
                y1, y2 = y_offset, y_offset + mute_icon_resized.shape[0]
                x1, x2 = x_offset, x_offset + mute_icon_resized.shape[1]

                # Split color and alpha channels
                icon_rgb = mute_icon_resized[:, :, :3]
                icon_alpha = (
                    mute_icon_resized[:, :, 3] / 255.0
                )  # Normalize alpha channel to 0-1

                # Blend each color channel based on alpha
                for c in range(3):  # Apply to B, G, R channels
                    debug_image[y1:y2, x1:x2, c] = (
                        icon_alpha * icon_rgb[:, :, c]
                        + (1 - icon_alpha) * debug_image[y1:y2, x1:x2, c]
                    )

        # Screen reflection #############################################################
        cv2.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv2.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width
        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


if __name__ == "__main__":
    main()
