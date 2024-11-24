#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
from dataclasses import dataclass
from enum import StrEnum
import itertools
from logging import info, warning
import logging
import os
from typing import Literal, Optional, override

import cv2
from cv2.typing import MatLike, Point, Rect
import numpy as np
from mediapipe.python.solutions import hands as mp_hands

from utils.cvfpscalc import CvFpsCalc
from utils.draw import (
    BoundingBoxType,
    draw_bounding_rect,
    draw_info,
    draw_info_text,
    draw_landmarks,
)

PUNCTUATION_MARKS = [".", ",", "?", "!", '"']

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class Mode(StrEnum):
    NORMAL = "NORMAL"
    KEYPOINT = "KEYPOINT_TRAINING"

    def is_normal(self):
        return self == Mode.NORMAL

    def is_keypoint(self):
        return self == Mode.KEYPOINT

    def pick_class_number(self, key: int):
        """If in keypoint training mode, returns the number pressed.
        Used to select what label to assign to keypoint data when in KEYPOINT_TRAINING mode.

        Returns:
            Optional[int]: The number pressed, or None if the mode is not KEYPOINT_TRAINING.
        """
        if not self.is_keypoint():
            return None

        if 48 <= key <= 57:
            return key - 48
        return None

    def log_data(self, number, landmark_list):
        if not self.is_keypoint() or number is None:
            return

        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])


MODE = Mode(os.getenv("MODE", "NORMAL"))


class ImageCaptureError(Exception):
    pass


@dataclass(frozen=True, eq=True, slots=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass
class HandResult:
    hand_landmarks: list[Landmark]
    handedness: Literal["Left"] | Literal["Right"]
    # hand_world_landmarks: list


class HandDetector(mp_hands.Hands):
    def __init__(self, **kwargs):
        kwargs["max_num_hands"] = 1
        super().__init__(**kwargs)

    @override
    def process(self, image: MatLike) -> Optional[HandResult]:
        image.flags.writeable = False
        results = super().process(image)
        image.flags.writeable = True

        multi_hand_landmarks = results.multi_hand_landmarks  # type:ignore
        multi_handedness = results.multi_handedness  # type:ignore
        multi_hand_world_landmarks = results.multi_hand_world_landmarks  # type:ignore

        if not (
            multi_hand_landmarks is not None
            and multi_handedness is not None
            and multi_hand_world_landmarks is not None
        ):
            return None

        assert len(multi_handedness) == 1

        return HandResult(
            hand_landmarks=multi_hand_landmarks[0].landmark,
            handedness=multi_handedness[0].classification[0].label,
            # hand_world_landmarks=multi_hand_world_landmarks[0].landmark,
        )


class UI:
    def __init__(self, image_width=960, image_height=540):
        pygame.init()
        pygame.mixer.init()
        self.window = pygame.display.set_mode((1024, 768))
        self.muted = False
        self.chime_file = "Chime.mp3"
        self.running = True
        self.paused = True
        self.mute_button = pygame.Rect(
            10 + image_width - 100, image_height + 50, 100, 50
        )
        self.start_stop_button = pygame.Rect(50, image_height + 50, 100, 50)
        self.text_box = pygame.Rect(50, image_height + 120, image_width - 40, 200)
        self.text = "Waiting to start recording..."
        self.image_width = image_width
        self.image_height = image_height
        
        self.font = pygame.font.Font(None, 36)
        self.font_height = self.font.get_height()

    def handle_start_stop_button(self):
        """Called when the start/stop button is clicked

        Toggles the paused state and updates button text,
        also will send a control signal to redis to tell the other
        components to start/stop processing

        TODO: implement redis stuff
        TODO: need a better way for the punctuation component to wait for the speech-to-text component to finish, it can't just run the inference as soon as it gets the stop signal

        """

        self.paused = not self.paused

        self.text = (
            "Recording..." if not self.paused else "Waiting to start recording..."
        )

        pass

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            info("Muted")
        else:
            info("Unmuted")

    def play_chime(self):
        def play_chime_thread(chime_file, muted):
            if not muted:
                pygame.mixer.music.load(chime_file)
                pygame.mixer.music.play()

        threading.Thread(
            target=play_chime_thread, args=(self.chime_file, self.muted)
        ).start()

    def process_events(self) -> int | None:
        """Process pygame events

        Returns:
            int | None: the keypoint class number if a number key was pressed and the mode is KEYPOINT_TRAINING, else None
        """
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    self.running = False
                    return
                case pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        return
                    if event.key == pygame.K_m:
                        self.toggle_mute()
                    if event.key == pygame.K_SPACE:
                        self.handle_start_stop_button()
                    if MODE.is_keypoint():
                        return MODE.pick_class_number(event.key)
                case pygame.MOUSEBUTTONDOWN:
                    if self.mute_button.collidepoint(event.pos):
                        self.toggle_mute()
                    if self.start_stop_button.collidepoint(event.pos):
                        self.handle_start_stop_button()

    def draw(self, image: MatLike):
        # TODO: the text box should:
        # - word wrap
        # - be able to scroll up and down
        # - have a thin border
        # - have an icon in the top right corner to copy the text to the clipboard
        # - text should be selectable

        image = cv2.transpose(image)
        display = pygame.surfarray.make_surface(image)

        self.window.fill((255, 255, 255))
        self.window.blit(display, (50, 50))

        # Draw text box 
        pygame.draw.rect(
            self.window, (102, 102, 255), self.text_box, border_radius=25
        )
        pygame.draw.rect(
            self.window, (0, 0, 0), self.text_box, 2, border_radius=25
        )
        self.window.blit(display, (50, 50))

        # Draw buttons with rounded edges and thin black border
        pygame.draw.rect(
            self.window, (255, 255, 255), self.mute_button, border_radius=10
        )
        pygame.draw.rect(
            self.window, (0, 0, 0), self.mute_button, 2, border_radius=10
        )
        pygame.draw.rect(
            self.window, (255, 255, 255), self.start_stop_button, border_radius=10
        )
        pygame.draw.rect(
            self.window, (0, 0, 0), self.start_stop_button, 2, border_radius=10
        )

        mute_text = self.font.render(
            "Unmute" if self.muted else "Mute", True, (0, 0, 0)
        )
        start_stop_text = self.font.render(
            "Start" if self.paused else "Stop", True, (0, 0, 0)
        )
        text = self.font.render(self.text, True, (0, 0, 0))

        mute_text_width = mute_text.get_width()
        start_stop_text_width = start_stop_text.get_width()

        self.window.blit(
            mute_text,
            (
                10 + self.image_width - 100 + (100 - mute_text_width) // 2,
                self.image_height + 50 + self.font_height // 2,
            ),
        )
        self.window.blit(
            start_stop_text,
            (
                50 + (100 - start_stop_text_width) // 2,
                self.image_height + 50 + self.font_height // 2,
            ),
        )
        self.window.blit(text, (70, self.image_height + 130 + 10))

        pygame.display.flip()


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
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    return args


def capture_image(cap: cv2.VideoCapture) -> MatLike:
    ret, image = cap.read()
    if not ret:
        raise ImageCaptureError("Failed to capture image")
    return image


def display_mute_icon(display: MatLike, muted: bool) -> MatLike:
    if not muted:
        return display

    try:
        mute_icon = cv2.imread("volume-mute.png", cv2.IMREAD_UNCHANGED)
    except cv2.error as e:
        warning(f"Failed to load mute icon: {e}")
        return display

    n = 70  # pixels
    mute_icon_resized = cv2.resize(
        mute_icon, (n, n)
    )  # Resize the mute icon to nxn pixels

    h, w, _ = display.shape  # Get the dimensions of the current frame
    x_offset, y_offset = w - (n + 10), h - (n + 10)  # Bottom-right corner
    y1, y2 = y_offset, y_offset + mute_icon_resized.shape[0]
    x1, x2 = x_offset, x_offset + mute_icon_resized.shape[1]

    # Split color and alpha channels
    icon_rgb = mute_icon_resized[:, :, :3]
    icon_alpha = mute_icon_resized[:, :, 3] / 255.0  # Normalize alpha channel to 0-1

    # Blend each color channel based on alpha
    for c in range(3):  # Apply to B, G, R channels
        display[y1:y2, x1:x2, c] = (
            icon_alpha * icon_rgb[:, :, c] + (1 - icon_alpha) * display[y1:y2, x1:x2, c]
        )

    return display


def main():
    # Set up logging
    logging.basicConfig(
        level=(
            logging._nameToLevel[LOG_LEVEL]
            if LOG_LEVEL in logging._nameToLevel
            else logging.INFO
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation #####################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    hand_detector = HandDetector(
        static_image_mode=use_static_image_mode,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    #  define redis queue #####################################################
    redis_connection = Redis(
        host="redis",
        # host="127.0.0.1",
        port=6379,
        retry_on_timeout=True,
        retry=Retry(backoff=ExponentialBackoff(), retries=3),
    )
    print("Connected to Redis!")

    # Define variables for tracking gestures ##################################
    gesture_detected = False
    gesture_start_time = time.time()
    dwell_time = 3  # 3 seconds
    keypoint_training_class: int | None = None

    # Set up UI ###############################################################
    ui = UI()

    try:
        while ui.running:
            fps = cvFpsCalc.get()

            # Process Events  ####################################################
            ui.process_events()

            # Camera capture #####################################################
            image = capture_image(cap)
            # the image that we process to detect the hand gestures
            image = cv2.flip(image, 1)  # Mirror display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # the image that we use to draw on to display stuff to the user
            display = copy.deepcopy(image)

            # Detection implementation #############################################################
            if (result := hand_detector.process(image)) is not None:
                hand_landmarks = result.hand_landmarks
                handedness = result.handedness

                # Bounding box calculation
                brect = calc_bounding_rect(display, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(display, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Write to the dataset file
                if MODE.is_keypoint() and keypoint_training_class is not None:
                    MODE.log_data(keypoint_training_class, pre_processed_landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Check if the detected gesture matches the desired punctuation ###################################
                # print(f"{keypoint_classifier_labels[hand_sign_id]}: {hand_sign_id}")
                if (
                    keypoint_classifier_labels[hand_sign_id] != "Neutral"
                    and not gesture_detected
                ):
                    # Start timing the gesture
                    gesture_detected = True
                    gesture_start_time = time.time()
                    # makes the rectangle orange as the dwell time is processing
                    display = draw_bounding_rect(
                        use_brect, display, brect, BoundingBoxType.Dwell
                    )
                elif (
                    keypoint_classifier_labels[hand_sign_id] != "Neutral"
                    and time.time() - gesture_start_time >= dwell_time
                ):
                    val_to_push = PUNCTUATION_MARKS[hand_sign_id]
                    # Dwell time met, push to queue only once
                    redis_connection.rpush("gesture_queue", val_to_push)
                    # makes the rectangle green when the val is queued successfully
                    display = draw_bounding_rect(
                        use_brect, display, brect, BoundingBoxType.Success
                    )
                    ui.play_chime()
                    print(f"Pushed {val_to_push} to queue.")

                    queued_val = redis_connection.lpop("gesture_queue")
                    if queued_val:
                        print(f"Dequeued {queued_val.decode()} from the queue.")
                        print()
                        # reset the reactangle  color back to normal
                        display = draw_bounding_rect(use_brect, display, brect)
                    gesture_detected = False  # Reset to prevent continuous queuing
                else:
                    # Reset if gesture is not detected in the frame
                    gesture_detected = False

                # Drawing part
                display = draw_bounding_rect(use_brect, display, brect)
                display = draw_landmarks(display, landmark_list)
                display = draw_info_text(
                    display,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                )

            display = draw_info(
                display, fps, MODE.is_keypoint(), keypoint_training_class
            )

            # Display the mute icon if the sound is muted #####################################
            display = display_mute_icon(display, ui.muted)

            # Screen reflection #############################################################
            # cv2.imshow("Hand Gesture Recognition", mat=display)
            ui.draw(display)
    except ImageCaptureError as e:
        warning(f"ImageCaptureError: {e}")
    finally:
        cap.release()
        pygame.quit()
        cv2.destroyAllWindows()


def calc_bounding_rect(image: MatLike, landmarks: list[Landmark]) -> Rect:
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    return cv2.boundingRect(landmark_array)


def calc_landmark_list(image, landmarks: list[Landmark]) -> list[Point]:
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for landmark in landmarks:
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


if __name__ == "__main__":
    main()
