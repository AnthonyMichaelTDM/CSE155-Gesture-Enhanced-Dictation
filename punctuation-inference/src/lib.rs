//! The punctuation-inference component takes in streams of WordEvents and GestureEvents and combines them to produce a properly punctuation text stream.
//!
//! ## Time Thresholds
//!
//! We receive word events *after* the word has been transcribed
//! and receive gesture events both when a gesture is first detected and when it is no longer detected.
//!
//! This means that we may be receiving gesture events meant for words that we don't have yet.
//!
//! As such, we can't just emit a punctuation event right away when we receive a gesture, we must wait until we have the word that the gesture is meant for.
//!
//! Let's look at some examples:
//!
//! (ascii chart)
//!                         end of word1
//!         |      hello       |   |         world       |
//! start time of word1         start of word2      end of word2
//!                                       end of gesture
//!    |                     !                      |
//!  start of gesture
//! ---------------------------------------------------------------------------> time
//!
//! - here, we receive the start of gesture, then word1, then the end of gesture, then word2
//! - we should emit "hello world!" after we receive word2, but how can we know that the gesture is meant for word2?
//!
//! (ascii chart)
//!                      end of word1
//!        |      hello       |   |         world       |
//! start time of word1         start of word2      end of word2
//!                               end of gesture
//!   |               !               |
//!
//! - here, we still receive the events in the same order, but the gesture is meant for word1 (so, expected output is "hello! world")
//! - how can we know that the gesture is meant for word1?
//!
//! To handle this, we need to keep track of the words and gestures that we have received so far and emit the punctuation when we have enough information to do so.
//! We will keep a queue of words and a queue of gestures and process them in order.
//! We will also keep track of the last gesture start event that we received.
//! We will emit the punctuation when we have a word and a gesture that are close enough in time.
//!
//! "close enough" is defined by some thresholds:
//!
//! - `WORD_END_GESTURE_START_TIME_THRESHOLD`: the maximum time difference between the end of a word and the end of a gesture for the gesture to placed immediately after the word
//! - `GESTURE_END_WORD_START_TIME_THRESHOLD`: the maximum time difference between the end of a gesture and the start of a word for the gesture to be placed immediately before the word
//! - `WORD_START_GESTURE_START_TIME_THRESHOLD`: the maximum time difference between the start of a word and the start of a gesture for the gesture to be placed sometime after the word
//!
//! - ``
//!
//!

mod events;

use std::collections::VecDeque;

use either::Either;

use crate::events::{GestureEvent, WordEvent};

/// Default time threshold for the maximum time difference between the end of a word and the end of a gesture for the gesture to be placed immediately after the word
/// This is useful for gestures that are detected after the word is fully transcribed
const WORD_END_GESTURE_START_TIME_THRESHOLD: f64 = 0.2;

/// Default time threshold for the maximum time difference between the end of a gesture and the start of a word for the gesture to be placed immediately before the word
/// This is useful for gestures that are detected before the word is fully transcribed
const GESTURE_END_WORD_START_TIME_THRESHOLD: f64 = 0.2;

/// Default time threshold for the maximum time difference between the start of a word and the start of a gesture for the gesture to be placed sometime after the word
/// This is useful for gestures that are detected before the word is fully transcribed
const WORD_START_GESTURE_START_TIME_THRESHOLD: f64 = 0.2;

pub trait PunctuationInference {
    fn process_word_event(&mut self, word: WordEvent);
    fn process_gesture_event(&mut self, gesture: GestureEvent);
    fn get_text(&self) -> String;
}

#[derive(Debug, Default)]
pub struct PunctuationInferenceBuilder {
    /// see: [`WORD_END_GESTURE_START_TIME_THRESHOLD`]
    word_end_gesture_end_threshold: Option<f64>,
    /// see: [`GESTURE_END_WORD_START_TIME_THRESHOLD`]
    gesture_end_word_start_threshold: Option<f64>,
    /// see: [`WORD_START_GESTURE_START_TIME_THRESHOLD`]
    word_start_gesture_start_threshold: Option<f64>,
}

impl PunctuationInferenceBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_word_end_gesture_end_threshold(
        mut self,
        word_end_gesture_end_threshold: f64,
    ) -> Self {
        self.word_end_gesture_end_threshold = Some(word_end_gesture_end_threshold);
        self
    }

    pub fn with_gesture_end_word_start_threshold(
        mut self,
        gesture_end_word_start_threshold: f64,
    ) -> Self {
        self.gesture_end_word_start_threshold = Some(gesture_end_word_start_threshold);
        self
    }

    pub fn with_word_start_gesture_start_threshold(
        mut self,
        word_start_gesture_start_threshold: f64,
    ) -> Self {
        self.word_start_gesture_start_threshold = Some(word_start_gesture_start_threshold);
        self
    }

    pub fn build(self) -> PunctuationInferenceImpl {
        PunctuationInferenceImpl {
            word_end_gesture_end_threshold: self
                .word_end_gesture_end_threshold
                .unwrap_or(WORD_END_GESTURE_START_TIME_THRESHOLD),
            gesture_end_word_start_threshold: self
                .gesture_end_word_start_threshold
                .unwrap_or(GESTURE_END_WORD_START_TIME_THRESHOLD),
            word_start_gesture_start_threshold: self
                .word_start_gesture_start_threshold
                .unwrap_or(WORD_START_GESTURE_START_TIME_THRESHOLD),

            output: Vec::new(),
            last_gesture_start: None,
            word_queue: VecDeque::new(),
            gesture_queue: VecDeque::new(),
        }
    }
}

/// Punctuation Inference
/// The Punctuation Inference component takes in streams of WordEvents and GestureEvents and combines them to produce a properly punctuated text stream.
#[derive(Debug)]
pub struct PunctuationInferenceImpl {
    /// the maximum time difference between the end of a word and the end of a gesture for the gesture to be placed after the word
    word_end_gesture_end_threshold: f64,
    /// the maximum time difference between the end of a gesture and the start of a word for the gesture to be placed before the word
    gesture_end_word_start_threshold: f64,
    /// the maximum time difference between the start of a word and the start of a gesture for the gesture to be placed after the word
    word_start_gesture_start_threshold: f64,
    /// Queue of words and punctuation that have been processed so far
    output: Vec<Either<WordEvent, GestureEvent>>,
    /// the last gesture start event that we received
    last_gesture_start: Option<GestureEvent>,
    /// The queue of words to be processed
    word_queue: VecDeque<WordEvent>,
    /// The queue of gestures to be processed
    gesture_queue: VecDeque<GestureEvent>,
}

impl PunctuationInference for PunctuationInferenceImpl {
    fn process_word_event(&mut self, word: WordEvent) {
        self.word_queue.push_back(word);
        self.process_queues();
    }

    fn process_gesture_event(&mut self, gesture: GestureEvent) {
        // if the gesture is an end gesture, we need to remove the corresponding start gesture from the queue before processing
        self.gesture_queue.push_back(gesture);
        self.process_queues();
    }

    fn get_text(&self) -> String {
        if self.output.is_empty() {
            return String::new();
        }
        if self.output.len() == 1 {
            match self.output.first() {
                Some(Either::Left(word)) => return word.word.clone(),
                _ => return String::new(),
            }
        }

        let mut result = self
            .output
            .windows(2)
            .map(|window| match window {
                // if we have a word followed by a gesture, we should combine them
                [Either::Left(word), Either::Right(gesture)] => {
                    word.word.clone() + &gesture.punctuation().to_string()
                }
                // if we have a gesture followed by a word, assume the gesture is meant for a previous word
                [Either::Right(_), Either::Left(_)] => String::new(),
                // if we have two words, we should just emit the first word
                [Either::Left(word), _] => word.word.clone(),
                // if we have two gestures, we should just emit the first gesture
                [Either::Right(gesture), _] => gesture.punctuation().to_string(),
                _ => unreachable!(),
            })
            .fold(String::new(), |acc, text| {
                if text.is_empty() {
                    acc
                } else if acc.is_empty() {
                    text
                } else {
                    acc + " " + &text
                }
            });

        // Handle the last element separately
        match self.output.last() {
            // if the last element is a word
            Some(Either::Left(word)) => {
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(&word.word);
            }
            _ => {}
        }

        result
    }
}

impl PunctuationInferenceImpl {
    /// Process the queues of words and gestures
    pub fn process_queues(&mut self) {
        match (self.word_queue.front(), self.gesture_queue.front()) {
            // if the gesture is a start gesture, we track it but don't emit it
            (_, Some(GestureEvent::Start { .. })) => {
                self.last_gesture_start = self.gesture_queue.pop_front();
            }
            // if we have an end gesture and no word, we wait
            (None, Some(GestureEvent::End { .. })) => {}

            // if we have a word and a gesture, but the gesture ends before the word starts, we skip that gesture
            // and try again
            (
                Some(word),
                Some(GestureEvent::End {
                    end_time: gesture_end_time,
                    ..
                }),
            ) if word.start_time > *gesture_end_time - self.gesture_end_word_start_threshold => {
                self.gesture_queue.pop_front();
                self.last_gesture_start = None;
                self.process_queues();
            }

            // if we have a word that starts within a threshold of the gesture, we emit that word's until the next word doesn't end within a threshold of the gesture end time, then emit the gesture
            (
                Some(word),
                Some(GestureEvent::End {
                    start_time: gesture_start_time,
                    end_time: gesture_end_time,
                    ..
                }),
            ) if word.start_time
                <= *gesture_start_time + self.word_start_gesture_start_threshold
                && word.end_time <= *gesture_end_time + self.word_end_gesture_end_threshold =>
            {
                let gesture_end_time = *gesture_end_time;
                while let Some(word) = self.word_queue.front() {
                    if word.end_time <= gesture_end_time + self.word_end_gesture_end_threshold {
                        self.emit_word();
                    } else {
                        break;
                    }
                }
                self.emit_gesture();
                self.last_gesture_start = None;
            }

            // if we have a word and no gesture has started yet, we emit the word
            (Some(_), None) if self.last_gesture_start.is_none() => {
                self.emit_word();
            }

            // if the queue is empty, we're done
            (_, _) => {}
        }
    }

    /// pop a word from the word queue into the output queue
    fn emit_word(&mut self) {
        if let Some(word) = self.word_queue.pop_front() {
            self.output.push(Either::Left(word.clone()));
        }
    }

    /// pop a gesture from the gesture queue into the output queue
    fn emit_gesture(&mut self) {
        if let Some(gesture) = self.gesture_queue.pop_front() {
            self.output.push(Either::Right(gesture.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[test]
    /// If we have no events, we should have no text
    fn test_no_events() {
        let punctuation_inference = PunctuationInferenceBuilder::new().build();
        assert_eq!(punctuation_inference.get_text(), "");
    }

    #[test]
    /// If we have only gestures, we should have no text
    /// This is because we can't infer punctuation without words
    fn test_only_gestures() {
        let mut punctuation_inference = PunctuationInferenceBuilder::new().build();
        punctuation_inference.process_gesture_event(GestureEvent::Start {
            punctuation: '!',
            start_time: 0.0,
        });
        punctuation_inference.process_gesture_event(GestureEvent::End {
            punctuation: '!',
            start_time: 0.0,
            end_time: 1.0,
            confidence: 0.9,
        });
        assert_eq!(punctuation_inference.get_text(), "");
    }

    #[rstest]
    /// If we have only words, we should have the text of the words
    /// This is because we can't infer punctuation without gestures
    #[case::one_word(vec![("hello", 0.0, 1.0, 0.9)], "hello")]
    #[case::two_words(vec![("hello", 0.0, 1.0, 0.9), ("world", 1.0, 2.0, 0.9)], "hello world")]
    fn test_only_words(#[case] words: Vec<(&str, f64, f64, f64)>, #[case] expected_text: &str) {
        let mut punctuation_inference = PunctuationInferenceBuilder::new().build();
        for (word, start_time, end_time, confidence) in words {
            punctuation_inference.process_word_event(WordEvent {
                word: word.to_string(),
                start_time,
                end_time,
                confidence,
            });
        }
        assert_eq!(punctuation_inference.get_text(), expected_text);
    }

    #[rstest]
    /// basic tests with one word and one gesture
    #[case::gesture_way_before_word(
            vec![Either::Right(GestureEvent::new_start('!', 0.0)),  Either::Right(GestureEvent::new_end('!', 0.0, 1.0, 0.9)), Either::Left(WordEvent::new("hello", 1.5, 2.0, 0.9))],
            "hello",
        )]
    #[case::gesture_way_after_word(
            vec![Either::Left(WordEvent::new("hello", 0.0, 1.0, 0.9)), Either::Right(GestureEvent::new_start('!', 1.5)), Either::Right(GestureEvent::new_end('!', 1.5, 2.0, 0.9))],
            "hello",
        )]
    #[case::gesture_overlaps_word(
            vec![Either::Right(GestureEvent::new_start('!', 0.0)), Either::Right(GestureEvent::new_end('!', 0.0, 1.0, 0.9)),Either::Left( WordEvent::new("hello", 0.0, 1.0, 0.9))],
            "hello!",
        )]
    #[case::gesture_overlaps_word(
            vec![Either::Right(GestureEvent::new_start('!', 0.0)), Either::Left( WordEvent::new("hello", 0.0, 1.0, 0.9)), Either::Right(GestureEvent::new_end('!', 0.0, 1.0, 0.9))],
            "hello!",
        )]
    #[case::gesture_start_before_word(
            vec![Either::Right(GestureEvent::new_start('!', 0.0)), Either::Left( WordEvent::new("hello", 0.1, 1.0, 0.9)), Either::Right(GestureEvent::new_end('!', 0.0, 1.0, 0.9))],
            "hello!",
        )]
    #[case::gesture_start_before_word(
            vec![Either::Right(GestureEvent::new_start('!', 0.0)), Either::Right(GestureEvent::new_end('!', 0.0, 0.2, 0.9)), Either::Left( WordEvent::new("hello", 0.1, 1.0, 0.9))],
            "hello",
        )]
    #[case::gesture_end_after_word(
            vec![Either::Right(GestureEvent::new_start('!', 0.0)), Either::Right(GestureEvent::new_end('!', 0.0, 1.0, 0.9)), Either::Left( WordEvent::new("hello", 0.9, 1.5, 0.9))],
            "hello",
        )]
    #[case::gesture_end_after_word(
            vec![Either::Right(GestureEvent::new_start('!', 0.0)), Either::Right(GestureEvent::new_end('!', 0.0, 1.4, 0.9)), Either::Left( WordEvent::new("hello", 0.1, 1.5, 0.9))],
            "hello!",
        )]
    fn test_one_word_one_gesture(
        #[case] events: Vec<Either<WordEvent, GestureEvent>>,
        #[case] expected_text: &str,
    ) {
        let mut punctuation_inference = PunctuationInferenceBuilder::new().build();
        // process the events in order
        for event in events {
            match event {
                Either::Left(word) => punctuation_inference.process_word_event(word),
                Either::Right(gesture) => punctuation_inference.process_gesture_event(gesture),
            }
        }

        assert_eq!(punctuation_inference.get_text(), expected_text);
    }
}
