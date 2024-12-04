use std::sync::mpsc;

use log::{debug, error, info, warn};
use punctuation_inference::{
    events::{ControlEvent, Event},
    redis::Redis,
    PunctuationInference, PunctuationInferenceBuilder,
};
use redis::ErrorKind;

fn main() -> anyhow::Result<()> {
    // initialize logging
    env_logger::init();

    let redis_port = std::env::var("REDIS_PORT").unwrap_or_else(|_| "6379".to_string());
    let redis_host = std::env::var("REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let redis_url = format!("redis://{}:{}", redis_host, redis_port);

    let (tx, rx) = mpsc::channel();

    let (stop_tx, stop_rx) = mpsc::channel();

    let handle = std::thread::spawn(move || {
        // try to connect to redis, retry on failure for 10 seconds
        let _redis = {
            let mut redis = None;
            for _ in 0..10 {
                match Redis::new(&redis_url) {
                    Ok(r) => {
                        redis = Some(r);
                        break;
                    }
                    Err(e) => {
                        error!("failed to connect to redis: {}", e);
                        std::thread::sleep(std::time::Duration::from_secs(1));
                    }
                }
            }

            redis.expect("failed to connect to redis")
        };

        info!("connected to redis");

        let _ = _redis.subscribe(&["word_events", "gesture_events", "control"], tx, stop_rx);
    });

    let mut punctuation_engine = PunctuationInferenceBuilder::default().build();
    let mut started = false;

    while let Ok(msg) = rx.recv() {
        let event = match msg.get_payload::<Event>() {
            Ok(event) => event,
            Err(e) if e.kind() == ErrorKind::TypeError => {
                warn!("received message with wrong type, skipping. Error: {e:?}. Message: {msg:?}");
                continue;
            }
            Err(e) => {
                error!("failed to get msg payload, exiting. Error: {e:?}. Message: {msg:?}");
                break;
            }
        };

        match event {
            Event::Word(word_event) if started => {
                info!("word event: {:?}", word_event);
                punctuation_engine.register_word_event(word_event);
            }
            Event::Gesture(gesture_event) if started => {
                info!("gesture event: {:?}", gesture_event);
                punctuation_engine.register_gesture_event(gesture_event);
            }
            Event::Control(ControlEvent::Start) => {
                info!("starting punctuation inference");
                started = true;
            }
            Event::Control(ControlEvent::StopRecording) => {
                started = false;
            }
            Event::Control(ControlEvent::Stop) => {
                info!("stopping punctuation inference and uploading results");
                started = false;

                // now we can run the engine and upload the results to redis
                let text = punctuation_engine.get_text();
                let mut redis = Redis::new("redis://redis:6379")?;
                redis.publish("punctuation_text", text.clone())?;
                info!("uploaded text: {text}");
            }
            Event::Control(ControlEvent::Reset) => {
                info!("resetting punctuation inference");
                punctuation_engine.reset();
            }
            Event::Control(ControlEvent::Exit) => {
                info!("exiting punctuation inference");
                break;
            }
            Event::Word(_) | Event::Gesture(_) => {
                warn!("received event before start event, skipping");
            }
            Event::Control(ControlEvent::Other) => {
                debug!("received unknown control event, skipping");
            }
        }
    }

    // at this point, we're done reading from redis, so we can send the kill signal
    stop_tx.send(()).expect("failed to send stop message");
    handle.join().expect("failed to join thread");

    Ok(())
}
