use crate::agent::Agent;
use crate::analyzer::{MarketStorage, analyze};
use crate::dqn_model::DqnModel;
use crate::env::Env;
use crate::replay_log::ReplaySample;
use crate::types::B;
use crate::websocket::{OrderBookData, TickData};

use burn::tensor::{Tensor, backend::Backend};
use tokio::sync::mpsc::Receiver;
use tokio::time::{Duration, timeout};

pub async fn run_trading_loop(
    agent: &mut Agent,
    model: &mut DqnModel<B>,
    env: &mut Env<B>,
    tick_receiver: &mut Receiver<TickData>,
    order_receiver: &mut Receiver<OrderBookData>,
    device: &<B as Backend>::Device,
) -> Vec<ReplaySample<B>> {
    let mut latest_order: Option<OrderBookData> = None;
    let mut storage = MarketStorage::new(200);
    let mut replay_batch: Vec<ReplaySample<B>> = Vec::new();

    while replay_batch.len() < 100 {
        if let Ok(Some(order)) = timeout(Duration::from_millis(10), order_receiver.recv()).await {
            latest_order = Some(order.clone());
            storage.push_orderbook(order);
        }

        if let Ok(Some(tick)) = timeout(Duration::from_millis(10), tick_receiver.recv()).await {
            storage.push_tick(tick.clone());

            if let Some(order) = &latest_order {
                if let Some(features) = analyze(&storage) {
                    env.update(features);

                    let state = env.observe();
                    let q_values = model.forward(state.clone());
                    let q_data = q_values.to_data().convert::<f32>();
                    let q_array = q_data.as_slice::<f32>().unwrap();
                    let action = agent.select_action(q_array);

                    let (next_state, reward) = env.step(action, tick.clone());

                    replay_batch.push(ReplaySample {
                        state,
                        action,
                        reward,
                        next_state,
                    });
                }
            }
        }
    }

    replay_batch
}
