#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import csv
import itertools
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from logging import error, info, warning
from turtle import st
from typing import TYPE_CHECKING, Callable, Literal, Optional, override

import cv2
import numpy as np
import pygame
from cv2.typing import MatLike, Point, Rect
from mediapipe.python.solutions import hands as mp_hands
from redis import Redis
from redis.backoff import ExponentialBackoff
from redis.client import PubSub
from redis.retry import Retry

from model import KeyPointClassifier
from utils.cvfpscalc import CvFpsCalc
from utils.draw import (
    BoundingBoxType,
    draw_bounding_rect,
    draw_info,
    draw_info_text,
    draw_landmarks,
)
from utils.events import (
    ControlEvent,
    GestureEndEvent,
    GestureEvent,
    GestureEventType,
    GestureStartEvent,
)

PUNCTUATION_MARKS = {
    "Period": ".",
    "Comma": ",",
    "Question Mark": "?",
    "Exclamation Point": "!",
    "Quotes": '"',
    "Neutral": "",
}

DWELL_TIME = 0.25  # second(s)
# minimum amount of time a gesture must be held to be considered a gesture
MIN_GESTURE_TIME = 0.05  # second(s)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

PUNCTUATED_TEXT_CHANNEL = "punctuation_text"
CONTROL_CHANNEL = "control"
GESTURE_CHANNEL = "gesture_events"


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


class RedisEventListener:
    """Listens for events on redis channels and calls the appropriate callback when an event is received."""

    @staticmethod
    def exception_handler(e, ps, worker):
        warning(f"Worker thread encountered an exception: {e}")
        worker.stop()

    def __init__(self, redis_conn: Redis):
        self.redis_conn: Redis = redis_conn
        self.pubsubs: dict[str, tuple[PubSub, Callable]] = {}
        self.pubsub_threads: dict[str, threading.Thread] = {}

    def subscribe(self, channel: str, callback: Callable[[str], None]):
        pubsub: PubSub = self.redis_conn.pubsub()
        pubsub.subscribe(**{channel: self._create_message_handler(callback)})
        self.pubsubs.update({channel: (pubsub, callback)})
        self.pubsub_threads[channel] = pubsub.run_in_thread(
            sleep_time=0.001,
            exception_handler=RedisEventListener.exception_handler,
        )

    def _create_message_handler(self, callback: Callable[[str], None]):
        def message_handler(message):
            if message["type"] == "message":
                callback(message["data"].decode("utf-8"))

        return message_handler

    def unsubscribe(self, channel: str):
        if channel not in self.pubsubs:
            return

        pubsub, _ = self.pubsubs[channel]
        pubsub.close()
        del self.pubsubs[channel]


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
    def __init__(self, redis_connection: Redis, image_width=960, image_height=540):
        pygame.init()
        pygame.mixer.init()
        self.window = pygame.display.set_mode((1024, 768))
        self.muted = False
        self.chime_file = "Chime.mp3"
        self.running = True
        self.paused = True
        self.recording_start_time = time.time()
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

        self.redis = redis_connection
        self.punctuated_text_listener = RedisEventListener(redis_connection)
        self.punctuated_text_listener.subscribe(
            PUNCTUATED_TEXT_CHANNEL, lambda x: self.puntuated_text_callback(x)
        )

    def puntuated_text_callback(self, text):
        if not self.paused:
            error("Received punctuated text while recording is still ongoing")

        self.text = text

    def handle_start_stop_button(self):
        """Called when the start/stop button is clicked

        Toggles the paused state and updates button text,
        also will send a control signal to redis to tell the other
        components to start/stop processing

        TODO: implement redis stuff
        TODO: need a better way for the punctuation component to wait for the speech-to-text component to finish, it can't just run the inference as soon as it gets the stop signal

        """

        self.paused = not self.paused

        if self.paused:
            self.redis.publish(CONTROL_CHANNEL, ControlEvent.stop_recording)
            self.text = "Recording stopped, please wait for processing to finish... "
        else:
            self.redis.publish(CONTROL_CHANNEL, ControlEvent.reset)
            self.redis.publish(CONTROL_CHANNEL, ControlEvent.start)
            self.text = "Recording..."
            self.recording_start_time = time.time()

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
        pygame.draw.rect(self.window, (102, 102, 255), self.text_box, border_radius=25)
        pygame.draw.rect(self.window, (0, 0, 0), self.text_box, 2, border_radius=25)
        self.window.blit(display, (50, 50))

        # Draw buttons with rounded edges and thin black border
        pygame.draw.rect(
            self.window, (255, 255, 255), self.mute_button, border_radius=10
        )
        pygame.draw.rect(self.window, (0, 0, 0), self.mute_button, 2, border_radius=10)
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
    bounding_box_type = BoundingBoxType.Default

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
    last_gesture_start: float | None = None
    last_gesture_detected: str | None = None
    keypoint_training_class: int | None = None
    start_sent = False

    # Set up UI ###############################################################
    ui = UI(redis_connection)

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

            # Preliminary displays (fps and mute icon) #####################################
            display = draw_info(
                display, fps, MODE.is_keypoint(), keypoint_training_class
            )
            display = display_mute_icon(display, ui.muted)

            # Detection implementation #############################################################
            result = hand_detector.process(image)

            if result is None:
                if (
                    last_gesture_start is not None
                    and time.time() - last_gesture_start < MIN_GESTURE_TIME
                ):
                    last_gesture_detected = None
                    last_gesture_start = None
                    bounding_box_type = BoundingBoxType.Default
                ui.draw(display)
                continue

            hand_landmarks = result.hand_landmarks
            handedness = result.handedness

            # Landmark calculation
            landmark_list = calc_landmark_list(display, landmarks=hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list, handedness
            )

            # Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

            # Extract the class of the keypoints
            detected_gesture_class = keypoint_classifier_labels[hand_sign_id]

            # Write to the dataset file if in keypoint training mode
            if MODE.is_keypoint() and keypoint_training_class is not None:
                MODE.log_data(keypoint_training_class, pre_processed_landmark_list)

            # landmark, bounding box, and info display #################################################################
            # Bounding box calculation
            brect = calc_bounding_rect(display, hand_landmarks)
            # check detected gesture against previous gesture to determine bounding box type
            if detected_gesture_class != last_gesture_detected:
                bounding_box_type = BoundingBoxType.Default
            elif (
                last_gesture_start is not None
                and time.time() - last_gesture_start >= DWELL_TIME
            ):
                bounding_box_type = BoundingBoxType.Success
            elif (
                last_gesture_start is not None
                and time.time() - last_gesture_start >= MIN_GESTURE_TIME
            ):
                bounding_box_type = BoundingBoxType.Dwell
            else:
                bounding_box_type = BoundingBoxType.Default

            display = draw_landmarks(display, landmark_list)
            display = draw_bounding_rect(use_brect, display, brect, bounding_box_type)
            display = draw_info_text(
                display,
                brect,
                handedness,
                keypoint_classifier_labels[hand_sign_id],
            )

            # Render GUI #############################################################
            ui.draw(display)

            # Gesture Event Sending ###################################
            # print(f"{keypoint_classifier_labels[hand_sign_id]}: {hand_sign_id}")

            event_to_send: Optional[GestureEvent] = None

            # some invariants
            assert not (
                last_gesture_detected is None and last_gesture_start is not None
            ), "last_gesture_detected should not be None if last_gesture_start is not None"
            assert not (
                last_gesture_detected is not None and last_gesture_start is None
            ), "last_gesture_start should not be None if last_gesture_detected is not None"
            assert last_gesture_detected != "Neutral", "we don't track neutral gestures"

            match (
                last_gesture_detected != detected_gesture_class,
                last_gesture_detected is None,
                detected_gesture_class == "Neutral",
            ):
                # we have no previous gesture, and the current gesture is neutral
                case (True, True, True):
                    pass
                # we have no previous gesture, and the current gesture is not neutral
                case (True, True, False):
                    # start timing the gesture
                    last_gesture_start = time.time()
                    last_gesture_detected = detected_gesture_class
                # we have a previous gesture (different from the current gesture), and the current gesture is neutral
                case (True, False, True):
                    if TYPE_CHECKING:
                        assert last_gesture_detected is not None
                    ## this happened before the minimum gesture time was reached
                    if (
                        last_gesture_start is not None
                        and time.time() - last_gesture_start <= MIN_GESTURE_TIME
                    ):
                        # clear the previous gesture
                        last_gesture_start = None
                        last_gesture_detected = None
                    ## this happened after the dwell time was reached
                    elif (
                        last_gesture_start is not None
                        and time.time() - last_gesture_start >= DWELL_TIME
                    ):
                        # we want to send the previous gesture
                        event_to_send = GestureEvent(
                            type=GestureEventType.End,
                            event=GestureEndEvent(
                                punctuation=PUNCTUATION_MARKS[last_gesture_detected],
                                start_time=max(
                                    0, last_gesture_start - ui.recording_start_time
                                ),
                                end_time=max(0, time.time() - ui.recording_start_time),
                                confidence=1.0,
                            ),
                        )
                        start_sent = False
                        last_gesture_start = None
                        last_gesture_detected = None
                    ## this happened after the minimum gesture time was reached, but before the dwell time was reached
                    else:
                        # ignore it, this was probably a mistake
                        pass
                # we have a previous gesture (different from the current gesture), and the current gesture is not neutral
                case (True, False, False):
                    if TYPE_CHECKING:
                        assert last_gesture_detected is not None
                    ## this happened before the minimum gesture time was reached
                    if (
                        last_gesture_start is not None
                        and time.time() - last_gesture_start <= MIN_GESTURE_TIME
                    ):
                        # we have a new gesture
                        last_gesture_start = time.time()
                        last_gesture_detected = detected_gesture_class
                    ## this happened after the dwell time was reached
                    elif (
                        last_gesture_start is not None
                        and time.time() - last_gesture_start >= DWELL_TIME
                    ):
                        # we have a new gesture, and we want to send the previous gesture
                        event_to_send = GestureEvent(
                            type=GestureEventType.End,
                            event=GestureEndEvent(
                                punctuation=PUNCTUATION_MARKS[last_gesture_detected],
                                start_time=max(
                                    0, last_gesture_start - ui.recording_start_time
                                ),
                                end_time=max(0, time.time() - ui.recording_start_time),
                                confidence=1.0,
                            ),
                        )
                        start_sent = False
                        last_gesture_start = time.time()
                        last_gesture_detected = detected_gesture_class
                    ## this happened after the minimum gesture time was reached, but before the dwell time was reached
                    else:
                        # ignore it, this was probably a mistake
                        pass
                # can't happen due to the invariants
                case (False, True, _):
                    pass
                case (False, _, True):
                    pass
                # we have a previous gesture (the same as the current gesture), and the current gesture is not neutral
                case (False, False, False):
                    if TYPE_CHECKING:
                        assert last_gesture_detected is not None
                    ## this happened after the minimum gesture time was reached, and we haven't sent the start event yet
                    if (
                        last_gesture_start is not None
                        and time.time() - last_gesture_start >= MIN_GESTURE_TIME
                        and not start_sent
                    ):
                        # we want to send the start event
                        event_to_send = GestureEvent(
                            type=GestureEventType.Start,
                            event=GestureStartEvent(
                                punctuation=PUNCTUATION_MARKS[detected_gesture_class],
                                start_time=max(
                                    0, last_gesture_start - ui.recording_start_time
                                ),
                            ),
                        )
                        start_sent = True
                    else:
                        pass

            if event_to_send is not None:
                if ui.paused:
                    warning(
                        f"Wanted to sent event {event_to_send}, but recording is paused"
                    )
                else:
                    redis_connection.publish(
                        GESTURE_CHANNEL, event_to_send.model_dump_json()
                    )

    except ImageCaptureError as e:
        warning(f"ImageCaptureError: {e}")
    finally:
        redis_connection.publish(CONTROL_CHANNEL, ControlEvent.exit)
        redis_connection.shutdown()
        redis_connection.close()
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


def pre_process_landmark(landmark_list: list[Point], handedness: str) -> list[float]:
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x  # type: ignore
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y  # type: ignore

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n: int) -> float:
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    if handedness == "Left":
        temp_landmark_list = [1.0 - x for x in temp_landmark_list]

    return temp_landmark_list


if __name__ == "__main__":
    main()
