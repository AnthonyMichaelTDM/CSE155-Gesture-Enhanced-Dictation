from dataclasses import dataclass
from enum import StrEnum
from logging import warning
from typing import Optional, Union

from pydantic import BaseModel, model_serializer


class ControlEvent(StrEnum):
    """
    A control event from the control channel
    """

    # Tell other components to start
    start = '"Start"'
    # Tell speech-to-text to stop recording audio and start transcribing
    stop_recording = '"Stop Recording"'
    # Tells the components to exit
    exit = '"Exit"'
    # Tells the punctuation inference component to clear its internal state
    reset = '"Reset"'

    @classmethod
    def from_str(cls, event_str: str) -> Optional["ControlEvent"]:
        try:
            return cls(event_str)
        except ValueError:
            warning(f"Invalid ControlEvent: {event_str}")
            return None


class GestureEventType(StrEnum):
    Start = "Start"
    End = "End"


@dataclass
class GestureStartEvent:
    """
    A gesture start event
    """

    punctuation: str
    start_time: float


@dataclass
class GestureEndEvent:
    """
    A gesture end event
    """

    punctuation: str
    start_time: float
    end_time: float
    confidence: float


class GestureEvent(BaseModel):
    type: GestureEventType
    event: Union[GestureStartEvent, GestureEndEvent]

    @model_serializer
    def serialize(self):
        match self.type:
            case GestureEventType.Start:
                return {
                    "Start": {
                        "punctuation": self.event.punctuation,
                        "start_time": self.event.start_time,
                    }
                }
            case GestureEventType.End:
                assert isinstance(self.event, GestureEndEvent)
                return {
                    "End": {
                        "punctuation": self.event.punctuation,
                        "start_time": self.event.start_time,
                        "end_time": self.event.end_time,
                        "confidence": self.event.confidence,
                    }
                }

    class Config:
        use_enum_values = True
