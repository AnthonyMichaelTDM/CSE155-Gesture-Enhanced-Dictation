use std::sync::mpsc;

use anyhow::Result;
use log::info;
use redis::RedisResult;
use redis::{Client, Connection, Msg};

use crate::events::{ControlEvent, Event};

#[allow(clippy::module_name_repetitions)]
pub struct Redis {
    _client: Client,
    connection: Connection,
}

impl Redis {
    pub fn new(url: &str) -> RedisResult<Self> {
        let client = Client::open(url)?;
        let connection = client.get_connection()?;

        Ok(Self {
            _client: client,
            connection,
        })
    }

    /// Subscribes to the given channels and listens to messages on the
    /// pubsub connection and sends them to the given sender
    ///
    /// will run as long as the channel is open and the sender is alive
    ///
    /// returns a one-shot channel that can be used to stop the subscription
    pub fn subscribe(
        mut self,
        channels: &[&str],
        tx: mpsc::Sender<Msg>,
        stop_rx: mpsc::Receiver<()>,
    ) -> Result<()> {
        let mut pubsub = self.connection.as_pubsub();
        for channel in channels {
            pubsub.subscribe(channel)?;
        }

        while let Ok(msg) = pubsub.get_message() {
            if matches!(
                msg.get_payload::<Event>(),
                Ok(Event::Control(ControlEvent::Exit))
            ) {
                tx.send(msg)?;
                break;
            }
            tx.send(msg)?;

            if stop_rx.try_recv().is_ok() {
                info!("stopping subscription");
                break;
            }
        }

        Ok(())
    }

    /// Publishes a message to the given channel
    pub fn publish(&mut self, channel: &str, message: impl AsRef<str>) -> RedisResult<()> {
        redis::cmd("PUBLISH")
            .arg(channel)
            .arg(message.as_ref())
            .query(&mut self.connection)
    }
}
