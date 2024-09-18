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
//! - here, we receive word1, then the start of gesture, then the end of gesture, then word2
//! - we should emit "hello world!" after we receive word2, but how can we know that the gesture is meant for word2?
//!
//! answer: we have some time thresholds:
//! - a threshold of time after a given word that, if surpassed by the time we get a gesture, we assume that the gesture is meant for the next word
//! - a threshold of time after a given gesture that, if surpassed by the time we get a word, we assume that the gesture is meant for the previous word
//!
//! the first threshold can handle the example above, but what about this one?
//!
//! (ascii chart)
//!                      end of word1
//!        |      hello       |   |         world       |
//! start time of word1         start of word2      end of word2
//!                               end of gesture
//!   |               !               |
//!
//! - here, we still receive the events in the same order, but the gesture is meant for word1
//! - although the end of the gesture came after the first threshold passed, we can tell that the gesture is meant for word1 because the time difference between the end of the gesture and the end of word2 is less than the second threshold
//!
//! the second threshold can handle this example
//!
//!
//! Another thing to consider is that although we can likely ignore a gesture that doesn't match any word, we must emit every word that we receive, even if we don't have a gesture for it yet.
//!
//!

mod events;

use std::collections::VecDeque;
use std::iter::Peekable;

use crate::events::{GestureEvent, WordEvent};

const DEFAULT_NEXT_WORD_THRESHOLD: f64 = 0.2;
const DEFAULT_PREVIOUS_WORD_THRESHOLD: f64 = 0.2;

pub trait PunctuationInference {
    fn process_word(&mut self, word: WordEvent);
    fn process_gesture(&mut self, gesture: GestureEvent);
    fn get_text(&self) -> String;
}

#[derive(Debug, Default)]
pub struct PunctuationInferenceBuilder {
    next_word_threshold: Option<f64>,
    previous_word_threshold: Option<f64>,
}

impl PunctuationInferenceBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the time threshold after a given word that, if surpassed by the time we get a gesture, we assume that the gesture is meant for the next word, default is 0.2
    pub fn with_next_word_threshold(mut self, next_word_threshold: f64) -> Self {
        self.next_word_threshold = Some(next_word_threshold);
        self
    }

    // Set the time threshold after a given gesture that, if surpassed by the time we get a word, we assume that the gesture is meant for the previous word, default is 0.2
    pub fn with_previous_word_threshold(mut self, previous_word_threshold: f64) -> Self {
        self.previous_word_threshold = Some(previous_word_threshold);
        self
    }

    pub fn build(self) -> PunctuationInferenceImpl {
        PunctuationInferenceImpl {
            text: String::new(),
            next_word_threshold: self
                .next_word_threshold
                .unwrap_or(DEFAULT_NEXT_WORD_THRESHOLD),
            previous_word_threshold: self
                .previous_word_threshold
                .unwrap_or(DEFAULT_PREVIOUS_WORD_THRESHOLD),
            word_queue: VecDeque::new(),
            gesture_queue: VecDeque::new(),
        }
    }
}

/// Punctuation Inference
/// The Punctuation Inference component takes in streams of WordEvents and GestureEvents and combines them to produce a properly punctuated text stream.
#[derive(Debug)]
pub struct PunctuationInferenceImpl {
    /// The current text segment being processed
    text: String,
    /// the time threshold after a given word that, if surpassed by the time we get a gesture, we assume that the gesture is meant for the next word
    next_word_threshold: f64,
    /// the time threshold after a given gesture that, if surpassed by the time we get a word, we assume that the gesture is meant for the previous word
    previous_word_threshold: f64,
    /// The queue of words to be processed
    word_queue: VecDeque<WordEvent>,
    /// The queue of gestures to be processed
    gesture_queue: VecDeque<GestureEvent>,
}

impl PunctuationInference for PunctuationInferenceImpl {
    fn process_word(&mut self, word: WordEvent) {
        self.word_queue.push_back(word);
        self.process_queues();
    }

    fn process_gesture(&mut self, gesture: GestureEvent) {
        // // if the gesture is an end gesture, we need to remove the corresponding start gesture from the queue before processing
        // if let GestureEvent::End {
        //     punctuation,
        //     start_time,
        //     end_time,
        //     confidence,
        // } = gesture
        // {
        //     let start_gesture = self.gesture_queue.iter().position(|g| match g {
        //         GestureEvent::Start {
        //             punctuation: p,
        //             start_time: s,
        //         } => *p == punctuation && *s == start_time,
        //         _ => false,
        //     });

        //     if let Some(index) = start_gesture {
        //         self.gesture_queue.remove(index);
        //     }
        // }
        self.gesture_queue.push_back(gesture);
        self.process_queues();
    }

    fn get_text(&self) -> String {
        std::iter::once(self.text.clone())
            .chain(self.word_queue.iter().map(|word| word.word.clone()))
            .filter(|s| !s.is_empty())
            .fold(String::new(), |acc, s| {
                if acc.is_empty() {
                    s
                } else {
                    acc + " " + &s
                }
            })
    }
}

impl PunctuationInferenceImpl {
    /// Process the queues of words and gestures
    pub fn process_queues(&mut self) {
        while let (Some(word), Some(_)) = (self.word_queue.front(), self.gesture_queue.front()) {
            self.text.push_str(&word.word);
            self.word_queue.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    #[rstest]
    #[case::default(
        PunctuationInferenceBuilder::new(),
        DEFAULT_NEXT_WORD_THRESHOLD,
        DEFAULT_PREVIOUS_WORD_THRESHOLD
    )]
    #[case::custom(PunctuationInferenceBuilder::new().with_next_word_threshold(0.5).with_previous_word_threshold(0.3), 0.5, 0.3)]
    #[case::custom(PunctuationInferenceBuilder::new().with_next_word_threshold(0.5), 0.5, DEFAULT_PREVIOUS_WORD_THRESHOLD)]
    #[case::custom(PunctuationInferenceBuilder::new().with_previous_word_threshold(0.3), DEFAULT_NEXT_WORD_THRESHOLD, 0.3)]
    fn test_punctuation_inference_builder(
        #[case] builder: PunctuationInferenceBuilder,
        #[case] next_word_threshold: f64,
        #[case] previous_word_threshold: f64,
    ) {
        let punctuation_inference = builder.build();
        assert_eq!(
            punctuation_inference.next_word_threshold,
            next_word_threshold
        );
        assert_eq!(
            punctuation_inference.previous_word_threshold,
            previous_word_threshold
        );
    }

    #[test]
    /// If we have no events, we should have no text
    fn test_no_events() {
        let mut punctuation_inference = PunctuationInferenceBuilder::new().build();
        assert_eq!(punctuation_inference.get_text(), "");
    }

    #[test]
    /// If we have only gestures, we should have no text
    /// This is because we can't infer punctuation without words
    fn test_only_gestures() {
        let mut punctuation_inference = PunctuationInferenceBuilder::new().build();
        punctuation_inference.process_gesture(GestureEvent::Start {
            punctuation: '!',
            start_time: 0.0,
        });
        punctuation_inference.process_gesture(GestureEvent::End {
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
            punctuation_inference.process_word(WordEvent {
                word: word.to_string(),
                start_time,
                end_time,
                confidence,
            });
        }
        assert_eq!(punctuation_inference.get_text(), expected_text);
    }
}
