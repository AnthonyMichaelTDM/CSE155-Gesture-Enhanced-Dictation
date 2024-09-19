use serde::{Deserialize, Serialize};

/// Word Events
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(untagged)]
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
