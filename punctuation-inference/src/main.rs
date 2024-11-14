use std::sync::mpsc;

use punctuation_inference::{
    events::{ControlEvent, Event},
    redis::Redis,
    PunctuationInference, PunctuationInferenceBuilder,
};

fn main() -> anyhow::Result<()> {
    let (tx, rx) = mpsc::channel();

    let (stop_tx, stop_rx) = mpsc::channel();

    let handle = std::thread::spawn(move || {
        // try to connect to redis, retry on failure for 10 seconds
        let _redis = {
            let mut redis = None;
            for _ in 0..10 {
                match Redis::new("redis://redis:6379") {
                    Ok(r) => {
                        redis = Some(r);
                        break;
                    }
                    Err(e) => {
                        eprintln!("failed to connect to redis: {}", e);
                        std::thread::sleep(std::time::Duration::from_secs(1));
                    }
                }
            }

            redis.expect("failed to connect to redis")
        };

        println!("connected to redis");

        let _ = _redis.subscribe(&["word_events", "gesture_events", "control"], tx, stop_rx);
    });

    let mut punctuation_engine = PunctuationInferenceBuilder::default().build();
    let mut started = false;

    while let Ok(msg) = rx.recv() {
        let event = msg.get_payload::<Event>()?;

        match event {
            Event::Word(word_event) if started => {
                println!("word event: {:?}", word_event);
                punctuation_engine.register_word_event(word_event);
            }
            Event::Gesture(gesture_event) if started => {
                println!("gesture event: {:?}", gesture_event);
                punctuation_engine.register_gesture_event(gesture_event);
            }
            Event::Control(ControlEvent::Start) => {
                println!("starting punctuation inference");
                started = true;
            }
            Event::Control(ControlEvent::Stop) => {
                println!("stopping punctuation inference");
                started = false;
            }
            Event::Control(ControlEvent::Reset) => {
                println!("resetting punctuation inference");
                punctuation_engine.reset();
            }
            Event::Control(ControlEvent::Exit) => {
                println!("exiting punctuation inference");
                break;
            }
            _ => {}
        }
    }

    // at this point, we're done reading from redis, so we can send the kill signal
    stop_tx.send(()).expect("failed to send stop message");
    handle.join().expect("failed to join thread");

    // now we can run the engine and upload the results to redis
    punctuation_engine.process_queues();
    let text = punctuation_engine.get_text();
    let mut redis = Redis::new("redis://redis:6379")?;
    redis.publish("punctuation_text", text)?;

    Ok(())
}
