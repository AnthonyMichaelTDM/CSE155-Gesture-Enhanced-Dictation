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
import tkinter as tk
from PIL import Image, ImageTk

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

    def log_data(self, number: int | None, landmark_list: list[tuple[float, float]]):
        if not self.is_keypoint() or number is None:
            return

        # convert to a one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(landmark_list))

        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *temp_landmark_list])


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
        pygame.mixer.init()
        self.root = tk.Tk()
        self.root.title("Gesture Enhanced Dictation")
        self.root.configure(bg="white")
        self.image_width = image_width
        self.image_height = image_height
        self.muted = False
        self.running = True
        self.paused = True
        self.recording_start_time = time.time()
        self.chime_file = "Chime.mp3"
        self.text = "Waiting to start recording..."

        self.redis = redis_connection
        self.punctuated_text_listener = RedisEventListener(redis_connection)
        self.punctuated_text_listener.subscribe(
            PUNCTUATED_TEXT_CHANNEL, lambda x: self.puntuated_text_callback(x)
        )
        self.last_number_keypress = None

        self.create_widgets()
        self.bind_events()

    def create_widgets(self):

        self.canvas = tk.Canvas(
            self.root, width=self.image_width, height=self.image_height, bg="white"
        )
        self.canvas.pack(padx=20, pady=20)

        self.button_frame = tk.Frame(self.root, bg="white")
        self.button_frame.pack(fill=tk.X, padx=20, pady=10)

        self.start_stop_button = tk.Button(
            self.button_frame,
            text="Start",
            command=self.handle_start_stop_button,
            bg="white",
            fg="black",
        )
        self.start_stop_button.pack(side=tk.LEFT, padx=5)

        self.mute_button = tk.Button(
            self.button_frame,
            text="Mute",
            command=self.toggle_mute,
            bg="white",
            fg="black",
        )
        self.mute_button.pack(side=tk.RIGHT, padx=5)

        self.text_box = tk.Text(self.root, height=10, width=50, bg="white", fg="black")
        self.text_box.pack(fill=tk.X, padx=20, pady=20)
        self.text_box.insert(tk.END, self.text)
        self.text_box.config(state=tk.DISABLED)  # Make the text box read-only

    def bind_events(self):
        self.root.bind("<space>", lambda _: self.handle_start_stop_button())
        self.root.bind("<m>", lambda _: self.toggle_mute())
        self.root.bind("<Escape>", lambda _: self.on_esc())
        for i in range(10):
            self.root.bind(str(i), self.on_number_keypress)

    def on_esc(self):
        self.running = False

    def on_number_keypress(self, event):
        if MODE.is_keypoint():
            number = int(event.char)

            self.last_number_keypress = (
                (number if number != self.last_number_keypress else None)
                if number is not None
                else self.last_number_keypress
            )

    def exit(self):
        self.root.quit()

    def update_text(self, text):
        self.text = text
        self.text_box.config(state=tk.NORMAL)
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, self.text)
        self.text_box.config(state=tk.DISABLED)

    def puntuated_text_callback(self, text):
        if not self.paused:
            error("Received punctuated text while recording is still ongoing")

        self.text = text

    def handle_start_stop_button(self):
        """Called when the start/stop button is clicked

        Toggles the paused state and updates button text,
        also will send a control signal to redis to tell the other
        components to start/stop processing
        """

        self.paused = not self.paused

        if self.paused:
            self.redis.publish(CONTROL_CHANNEL, ControlEvent.stop_recording)
            self.update_text(
                "Recording stopped, please wait for processing to finish... "
            )
            self.start_stop_button.config(text="Start")
        else:
            self.redis.publish(CONTROL_CHANNEL, ControlEvent.reset)
            self.redis.publish(CONTROL_CHANNEL, ControlEvent.start)
            self.update_text("Recording...")
            self.start_stop_button.config(text="Stop")
            self.recording_start_time = time.time()

        pass

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.mute_button.config(text="Unmute")
            info("Muted")
        else:
            self.mute_button.config(text="Mute")
            info("Unmuted")

    def play_chime(self):
        def play_chime_thread(chime_file, muted):
            if not muted:
                pygame.mixer.music.load(chime_file)
                pygame.mixer.music.play()

        threading.Thread(
            target=play_chime_thread, args=(self.chime_file, self.muted)
        ).start()

    def draw(self, image: MatLike):
        # TODO: the text box should:
        # - have an icon in the top right corner to copy the text to the clipboard

        display = Image.fromarray(image)
        display = ImageTk.PhotoImage(display)

        self.update_text(self.text)  # workaround to avoid updating text within a thread
        self.canvas.create_image(0, 0, image=display, anchor=tk.NW)
        self.root.update_idletasks()
        self.root.update()


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
    last_gesture_confidence: float | None = None
    keypoint_training_class: int | None = None
    start_sent = False

    # Set up UI ###############################################################
    ui = UI(redis_connection)

    try:
        while ui.running:
            fps = cvFpsCalc.get()

            # Process Events  ####################################################
            keypoint_training_class = ui.last_number_keypress

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
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Hand sign classification
            hand_sign_id, confidence = keypoint_classifier(pre_processed_landmark_list)

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
                confidence,
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
                    last_gesture_confidence = confidence
                # we have a previous gesture (different from the current gesture), and the current gesture is neutral
                case (True, False, True):
                    if TYPE_CHECKING:
                        assert last_gesture_detected is not None
                        assert last_gesture_confidence is not None
                    ## this happened before the minimum gesture time was reached
                    if (
                        last_gesture_start is not None
                        and time.time() - last_gesture_start <= MIN_GESTURE_TIME
                    ):
                        # clear the previous gesture
                        last_gesture_start = None
                        last_gesture_detected = None
                        last_gesture_confidence = None
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
                                confidence=last_gesture_confidence,
                            ),
                        )
                        start_sent = False
                        last_gesture_start = None
                        last_gesture_detected = None
                        last_gesture_confidence = None
                    ## this happened after the minimum gesture time was reached, but before the dwell time was reached
                    else:
                        # ignore it, this was probably a mistake
                        pass
                # we have a previous gesture (different from the current gesture), and the current gesture is not neutral
                case (True, False, False):
                    if TYPE_CHECKING:
                        assert last_gesture_detected is not None
                        assert last_gesture_confidence is not None
                    ## this happened before the minimum gesture time was reached
                    if (
                        last_gesture_start is not None
                        and time.time() - last_gesture_start <= MIN_GESTURE_TIME
                    ):
                        # we have a new gesture
                        last_gesture_start = time.time()
                        last_gesture_detected = detected_gesture_class
                        last_gesture_confidence = confidence
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
                                confidence=confidence,
                            ),
                        )
                        start_sent = False
                        last_gesture_start = time.time()
                        last_gesture_detected = detected_gesture_class
                        last_gesture_confidence = confidence
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
                        last_gesture_confidence = max(
                            last_gesture_confidence or 0, confidence
                        )
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


def pre_process_landmark(landmark_list: list[Point]) -> list[tuple[float, float]]:
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    for landmark_point in temp_landmark_list:
        landmark_point[0] = landmark_point[0] - base_x  # type: ignore
        landmark_point[1] = landmark_point[1] - base_y  # type: ignore

    # Normalization
    # this isn't the best way to do this, but its necessary so we can reuse our old data
    max_value = max(list(map(lambda x: max(abs(x[0]), abs(x[1])), temp_landmark_list)))

    def normalize_(n: Point) -> tuple[float, float]:
        return (n[0] / max_value, n[1] / max_value)

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


if __name__ == "__main__":
    main()
