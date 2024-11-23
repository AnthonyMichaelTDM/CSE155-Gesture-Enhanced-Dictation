import logging
import os
import string
import wave
from enum import Enum, StrEnum
from logging import debug, error, info, warning
from time import sleep
from typing import Callable, Generator, NoReturn, Optional

import pyaudio
import sounddevice as sd
import stable_whisper
import torch
from pydantic import BaseModel
from redis import Redis
from redis.backoff import ExponentialBackoff
from redis.client import PubSub
from redis.retry import Retry

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# variable to tell if the application is running in production (should wait for a redis connection)
PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"
CONTROL_CHANNEL = "control"
WORD_EVENTS_CHANNEL = "word_events"

# debug flag that can be set to skip recording audio and instead use a pre-recorded file
SKIP_RECORDING = bool(os.getenv("SKIP_RECORDING", False))
PRE_RECORDED_TEST_AUDIO = "harvard.wav"

AUDIO_FILE = "audio.wav"

HAS_GPU = torch.cuda.is_available() and torch.cuda.device_count() > 0
MODEL = "tiny.en"
LANGUAGE = "en"


class WordEvent(BaseModel):
    """
    A word event sent to the word_events channel
    """

    word: str
    start_time: float
    end_time: float
    confidence: float


class ControlEvent(StrEnum):
    """
    A control event from the control channel
    """

    start = '"Start"'
    stop_recording = '"Stop Recording"'
    exit = '"Exit"'

    @classmethod
    def from_str(cls, event_str: str) -> Optional["ControlEvent"]:
        try:
            return cls(event_str)
        except ValueError:
            warning(f"Invalid ControlEvent: {event_str}")
            return None


class RedisEventListener:
    """Listens for events on redis channels and calls the appropriate callback when an event is received."""

    @staticmethod
    def exception_handler(e, ps, worker):
        warning(f"Worker thread encountered an exception: {e}")
        worker.stop()

    def __init__(self, redis_conn: Redis):
        self.redis_conn: Redis = redis_conn
        self.pubsubs: dict[str, tuple[PubSub, Callable]] = {}

    def subscribe(self, channel: str, callback: Callable[[str], None]):
        pubsub: PubSub = self.redis_conn.pubsub()
        pubsub.subscribe(**{channel: self._create_message_handler(callback)})
        self.pubsubs.update({channel: (pubsub, callback)})
        self.pubsub_thread = pubsub.run_in_thread(
            sleep_time=0.001,
            exception_handler=RedisEventListener.exception_handler,
        )

    def _create_message_handler(self, callback: Callable[[str], None]):
        def message_handler(message):
            if message["type"] == "message":
                callback(message["data"].decode("utf-8"))

        return message_handler


class AudioTranscriber:
    def __init__(self):
        self.model = stable_whisper.load_faster_whisper(
            model_size_or_path=MODEL,
            device="cuda" if HAS_GPU else "cpu",
        )
        # self.word_event_generator: Optional[Generator[WordEvent, None, None]] = None

    def __report_progress(self, processed: float, total: float):
        debug(f"Processed {processed}/{total}s ({processed/total*100:.2f}%) of audio")

    # def start_transcription(self, audio_path: str, callback: Callable[[WordEvent], None]) -> Generator[WordEvent, None, None]:
    def start_transcription(self, audio_path: str) -> Generator[WordEvent, None, None]:
        # stable_whisper.whisper_word_level.faster_whisper.faster_transcribe(
        #     model=self.model,
        #     audio_path=audio_path,
        #     language=LANGUAGE,
        #     word_timestamps=True,
        #     vad=True
        # )

        segments, transcription_info = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language=LANGUAGE,
            prepend_punctuations="",
            append_punctuations="",
            vad_filter=True,
        )

        for segment in segments:
            # report progress
            self.__report_progress(segment.end, transcription_info.duration)

            if not segment.words:
                continue

            for word in segment.words:
                # ignore punctuation
                if any(char in string.punctuation for char in word.word):
                    debug(f"Ignoring punctuation: {word.word}")
                    continue

                word_text = word.word.strip()

                if not word_text:
                    continue

                word_event = WordEvent(
                    word=word_text,
                    start_time=word.start,
                    end_time=word.end,
                    confidence=word.probability,
                )
                yield word_event
        return


class AudioRecorder:
    def __init__(
        self, sample_rate: int = 44100, chunk_size: int = 1024, channels: int = 1
    ):
        self.p = pyaudio.PyAudio()
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.channels: int = channels
        self.stream: pyaudio._Stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        self.frames: list[bytes] = []

        # ensure stream is stopped
        if not self.stream.is_stopped():
            self.stream.stop_stream()

    def __del__(self):
        """
        Ensure the stream is closed when the object is deleted
        """
        if hasattr(self, "stream"):
            if not self.stream.is_stopped():
                self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, "p"):
            self.p.terminate()

    def record_frame(self):
        """
        Record a frame of audio
        """
        frame = self.stream.read(self.chunk_size)
        self.frames.append(frame)

    def save_audio(self, filename: str):
        if SKIP_RECORDING:
            warning("Skipping saving audio")
            return

        if not self.frames:
            warning("No audio to save")
            return
        if not self.stream.is_stopped():
            warning("Stream is still running")

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))
        self.frames = []


class _State(Enum):
    """
    The state of the state machine
    """

    EXIT = 0
    IDLE = 1
    RECORDING = 2
    TRANSCIBING = 3


class StateMachine:
    """
    State machine to control the application
    """

    def __init__(
        self,
        audio_recorder: AudioRecorder,
        transcriber: AudioTranscriber,
        redis: Optional[Redis] = None,
    ):
        self.state = _State.IDLE
        self.recorder = audio_recorder
        self.transcriber = transcriber
        self.redis = redis
        self.word_event_generator: Optional[Generator[WordEvent, None]] = None

        # if not running in production, start recording immediately
        if not PRODUCTION:
            self.control_event_handler(ControlEvent.start)
            # if SKIP_RECORDING is true, then we should start transcribing the pre-recorded audio immediately
            if SKIP_RECORDING:
                self.control_event_handler(ControlEvent.stop_recording)

    def control_event_handler(self, event: ControlEvent):
        match event:
            case ControlEvent.start:
                if self.state != _State.IDLE:
                    error("Received start event while not in IDLE state")
                    return
                info("Received start event")
                # start recording audio
                self.state = _State.RECORDING
                if not SKIP_RECORDING:
                    self.recorder.stream.start_stream()
            case ControlEvent.stop_recording:
                info("Received stop recording event")
                if self.state != _State.RECORDING:
                    warning(
                        "Received stop recording event while not in RECORDING state"
                    )
                    return
                # save audio to a file and send it to the model
                self.state = _State.TRANSCIBING
                self.recorder.stream.stop_stream()

                if not len(self.recorder.frames) == 0:
                    # save audio to a file
                    self.recorder.save_audio(AUDIO_FILE)
                else:
                    warning(
                        "Received stop recording event while no audio has been recorded"
                    )

                # transcribe audio
                self.word_event_generator = self.transcriber.start_transcription(
                    audio_path=(
                        AUDIO_FILE if not SKIP_RECORDING else PRE_RECORDED_TEST_AUDIO
                    )
                )

            case ControlEvent.exit:
                info("Received exit event")
                self.state = _State.EXIT
                # Halt any in-progress recording or transcription
                if not self.recorder.stream.is_stopped():
                    warning(
                        "Received exit event while stream is still running (no audio will be transcribed)"
                    )
                    self.recorder.stream.stop_stream()
                    self.recorder.stream.close()
                    self.recorder.p.terminate()
                # TODO: halt transcription

    def run(self) -> NoReturn:
        while True:
            match self.state:
                case _State.IDLE:
                    # debug("Waiting for start event")
                    sleep(0.05)
                case _State.RECORDING:
                    # debug("Recording audio")
                    # record audio
                    if not SKIP_RECORDING:
                        self.recorder.record_frame()
                    else:
                        sleep(0.05)
                case _State.TRANSCIBING:
                    # debug("Transcribing audio")

                    # send word events to the word_events channel
                    if self.word_event_generator is not None:
                        # take the next word event
                        try:
                            word_event = next(self.word_event_generator)
                            info(f"Transcribed word: {word_event.word}")

                            # send word event to redis
                            if self.redis is not None:
                                self.redis.publish(
                                    WORD_EVENTS_CHANNEL, word_event.model_dump_json()
                                )
                        except StopIteration:
                            info("Transcription complete")
                            self.state = _State.IDLE
                            self.word_event_generator = None
                            # if we're not running with redis, then exit after transcription is complete
                            if not PRODUCTION:
                                self.control_event_handler(ControlEvent.exit)
                case _State.EXIT:
                    info("Exiting application")
                    if self.redis is not None:
                        self.redis.close()

                    exit(0)


def list_devices():
    """print all available audio devices"""
    devices = sd.query_devices()
    print(f"Audio Devices:\n{devices}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=(
            logging._nameToLevel[LOG_LEVEL]
            if LOG_LEVEL in logging._nameToLevel
            else logging.INFO
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    list_devices()

    # connect to redis server
    redis_conn: Optional[Redis] = None
    if PRODUCTION:
        redis_conn = Redis(
            host="redis",
            # host="127.0.0.1",
            port=6379,
            retry_on_timeout=True,
            retry=Retry(backoff=ExponentialBackoff(), retries=10),
        )
        info("Connected to redis")

    # initialize state machine
    audio_recorder = AudioRecorder()
    transcriber = AudioTranscriber()
    app = StateMachine(
        audio_recorder=audio_recorder, transcriber=transcriber, redis=redis_conn
    )

    info("Application Started, waiting for events")

    # attach event listeners
    if PRODUCTION:
        assert redis_conn is not None
        listener = RedisEventListener(redis_conn)

        listener.subscribe(
            CONTROL_CHANNEL,
            lambda x: (
                app.control_event_handler(event)
                if (event := ControlEvent.from_str(x)) is not None
                else None
            ),
        )

    try:
        app.run()
    except KeyboardInterrupt:
        info("Received keyboard interrupt")
        exit(0)
    except Exception as e:
        error(f"An unhandled exception occurred: {e}")
        exit(1)
