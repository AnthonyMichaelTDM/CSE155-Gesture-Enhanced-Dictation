use redis::FromRedisValue;
use serde::{Deserialize, Serialize};

/// Event
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Event {
    Word(WordEvent),
    Gesture(GestureEvent),
    Control(ControlEvent),
}

impl FromRedisValue for Event {
    fn from_redis_value(v: &redis::Value) -> redis::RedisResult<Self> {
        let event: Event = serde_json::from_str(&String::from_redis_value(v)?).map_err(|_| {
            redis::RedisError::from((
                redis::ErrorKind::TypeError,
                "failed to parse event from redis value",
            ))
        })?;
        Ok(event)
    }
}

/// Control Events
/// These events are used to control the punctuation inference component
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlEvent {
    /// Start the punctuation inference process
    Start,
    /// Stop the punctuation inference process and upload the results
    Stop,
    /// Reset the punctuation inference process
    Reset,
    /// Exit the punctuation inference process (stop execution)
    Exit,
    /// Other, unknown control event
    #[serde(other)]
    Other,
}

/// Word Events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WordEvent {
    /// the word that was transcribed
    pub word: String,
    /// start time of the word in seconds, relative to the start of the audio file
    pub start_time: f64,
    /// end time of the word in seconds, relative to the start of the audio file
    pub end_time: f64,
    /// confidence level of the transcription, between 0 and 1
    pub confidence: f64,
}

impl WordEvent {
    pub fn new<S: ToString>(word: S, start_time: f64, end_time: f64, confidence: f64) -> Self {
        Self {
            word: word.to_string(),
            start_time,
            end_time,
            confidence,
        }
    }
}

/// Gesture Events
/// What this means is that when a gesture is first detected, end_time and confidence will
/// not be present, when the gesture is no longer detected, end_time and confidence
/// will be present
///
/// examples of the 2 variants:
/// ```json
/// {
///   "punctuation": "!",
///   "start_time": 0.0
/// }
/// ```
/// ```json
/// {
///   "punctuation": "!",
///   "start_time": 0.0,
///   "end_time": 1.0,
///   "confidence": 0.9
/// }
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub enum GestureEvent {
    Start {
        /// the punctuation character associated with the gesture that was detected
        punctuation: char,
        /// start time of the gesture in seconds, relative to the start of the video file
        start_time: f64,
    },
    End {
        /// the punctuation character associated with the gesture that was detected
        punctuation: char,
        /// start time of the gesture in seconds, relative to the start of the video file
        start_time: f64,
        /// end time of the gesture in seconds, relative to the start of the video file
        end_time: f64,
        /// average confidence level of the gesture detection, between 0 and 1
        confidence: f64,
    },
}

impl GestureEvent {
    pub fn new_start(punctuation: char, start_time: f64) -> Self {
        Self::Start {
            punctuation,
            start_time,
        }
    }

    pub fn new_end(punctuation: char, start_time: f64, end_time: f64, confidence: f64) -> Self {
        Self::End {
            punctuation,
            start_time,
            end_time,
            confidence,
        }
    }

    pub fn start_time(&self) -> f64 {
        match self {
            GestureEvent::Start { start_time, .. } => *start_time,
            GestureEvent::End { start_time, .. } => *start_time,
        }
    }

    pub fn punctuation(&self) -> char {
        match self {
            GestureEvent::Start { punctuation, .. } => *punctuation,
            GestureEvent::End { punctuation, .. } => *punctuation,
        }
    }
}

/// Text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Text {
    // the text segment with the punctuation inferred by the Punctuation Inference component
    pub text: String,
}

#[cfg(test)]
mod serialization_tests {
    //! Tests that document the format that the events are expected to be in
    //! when they are received from the redis pubsub channel

    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[rstest]
    #[case::word_event(
        r#"{"word":"hello","start_time":0.0,"end_time":1.0,"confidence":0.9}"#,
        Event::Word(WordEvent {
            word: "hello".to_string(),
            start_time: 0.0,
            end_time: 1.0,
            confidence: 0.9
        })
    )]
    #[case::gesture_event_start(
        r#"{"Start":{"punctuation":"!","start_time":0.0}}"#,
        Event::Gesture(GestureEvent::Start {
            punctuation: '!',
            start_time: 0.0
        })
    )]
    #[case::gesture_event_end(
        r#"{"End":{"punctuation":"!","start_time":0.0,"end_time":1.0,"confidence":0.9}}"#,
        Event::Gesture(GestureEvent::End {
            punctuation: '!',
            start_time: 0.0,
            end_time: 1.0,
            confidence: 0.9
        })
    )]
    #[case::control_event_start(r#""Start""#, Event::Control(ControlEvent::Start))]
    #[case::control_event_stop(r#""Stop""#, Event::Control(ControlEvent::Stop))]
    #[case::control_event_reset(r#""Reset""#, Event::Control(ControlEvent::Reset))]
    #[case::control_event_exit(r#""Exit""#, Event::Control(ControlEvent::Exit))]
    fn test_deserialization(#[case] value: &str, #[case] expected: Event) {
        let event: Event = serde_json::from_str(&value).unwrap();
        assert_eq!(event, expected);
    }

    #[rstest]
    #[case::word_event(
        Event::Word(WordEvent {
            word: "hello".to_string(),
            start_time: 0.0,
            end_time: 1.0,
            confidence: 0.9
        }),
        r#"{"word":"hello","start_time":0.0,"end_time":1.0,"confidence":0.9}"#
    )]
    #[case::gesture_event_start(
        Event::Gesture(GestureEvent::Start {
            punctuation: '!',
            start_time: 0.0
        }),
        r#"{"Start":{"punctuation":"!","start_time":0.0}}"#,
    )]
    #[case::gesture_event_end(
        Event::Gesture(GestureEvent::End {
            punctuation: '!',
            start_time: 0.0,
            end_time: 1.0,
            confidence: 0.9
        }),
        r#"{"End":{"punctuation":"!","start_time":0.0,"end_time":1.0,"confidence":0.9}}"#,
    )]
    #[case::control_event_start(Event::Control(ControlEvent::Start), r#""Start""#)]
    #[case::control_event_stop(Event::Control(ControlEvent::Stop), r#""Stop""#)]
    #[case::control_event_reset(Event::Control(ControlEvent::Reset), r#""Reset""#)]
    #[case::control_event_exit(Event::Control(ControlEvent::Exit), r#""Exit""#)]
    fn test_serialization(#[case] event: Event, #[case] expected: &str) {
        let value = serde_json::to_string(&event).unwrap();
        assert_eq!(value, expected);
    }
}
