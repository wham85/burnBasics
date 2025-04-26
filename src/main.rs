mod agent;
mod analyzer;
mod dqn_model;
mod env;
mod replay_log;
mod replaybuffer;
mod train;
mod types;
mod websocket;
mod model_saver; // 추가

use crate::types::B;
use agent::Agent;
use analyzer::{MarketStorage, analyze};
use burn::tensor::backend::Backend;
use csv::Writer;
use dqn_model::DqnModel;
use env::Env;
use replay_log::ReplaySample;
use std::fs::File;
use std::io::{self, Write};
use tokio::sync::mpsc;
use tokio::time::{Duration, timeout};
use train::train_from_csv;
use websocket::{OrderBookData, TickData, upbit_websocket_handler};
use model_saver::{save_model, load_model}; // 추가

#[tokio::main]
async fn main() {
    let coin = get_coin_symbol();

    let (tick_sender, mut tick_receiver) = mpsc::channel::<TickData>(100);
    let (order_sender, mut order_receiver) = mpsc::channel::<OrderBookData>(100);

    tokio::spawn(upbit_websocket_handler(
        coin.clone(),
        tick_sender,
        order_sender,
    ));

    let device = <B as Backend>::Device::default();
    let model_path = "models/latest_model.mpk";

    let mut model = if std::path::Path::new(model_path).exists() {
        load_model(model_path, &device)
    } else {
        DqnModel::new(&device)
    };

    let mut agent = Agent::new(0.9);
    let mut env = Env::new(device.clone());
    let mut storage = MarketStorage::new(200);
    let mut latest_order: Option<OrderBookData> = None;
    let mut replay_batch: Vec<ReplaySample> = Vec::new();
    let mut batch_index = 0;

    loop {
        if let Ok(Some(order)) = timeout(Duration::from_millis(10), order_receiver.recv()).await {
            latest_order = Some(order.clone());
            storage.push_orderbook(order);
        }

        if let Ok(Some(tick)) = timeout(Duration::from_millis(10), tick_receiver.recv()).await {
            storage.push_tick(tick.clone());

            if latest_order.is_some() {
                if let Some(features) = analyze(&storage) {
                    env.update(features);

                    let state = env.observe();
                    let q_values = model.forward(state.clone());
                    let q_array = q_values
                        .to_data()
                        .convert::<f32>().clone();
                    let q_array = q_array
                        .as_slice::<f32>()
                        .unwrap();
                    let action = agent.select_action(q_array);
                    let (next_state, reward) = env.step(action, tick.clone());

                    replay_batch.push(ReplaySample {
                        state,
                        action,
                        reward,
                        next_state,
                    });

                    if replay_batch.len() >= 100 {
                        let filename = format!("replay_{}.csv", batch_index);
                        save_replay_csv(&replay_batch, &filename);
                        train_from_csv(&filename, &mut model);
                        save_model(&model, model_path);
                        replay_batch.clear();
                        batch_index += 1;
                    }
                }
            }
        }
    }
}

fn get_coin_symbol() -> String {
    print!("💬 구독할 코인 심볼을 입력하세요 (예: KRW-BTC): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn save_replay_csv(batch: &[ReplaySample], filename: &str) {
    #[derive(serde::Serialize)]
    struct Row {
        action: usize,
        reward: f32,
        state_0: f32,
        state_1: f32,
        state_2: f32,
        state_3: f32,
        state_4: f32,
        state_5: f32,
        state_6: f32,
        state_7: f32,
        state_8: f32,
        state_9: f32,
        state_10: f32,
        state_11: f32,
        next_0: f32,
        next_1: f32,
        next_2: f32,
        next_3: f32,
        next_4: f32,
        next_5: f32,
        next_6: f32,
        next_7: f32,
        next_8: f32,
        next_9: f32,
        next_10: f32,
        next_11: f32,
    }

    let file = File::create(filename).unwrap();
    let mut writer = Writer::from_writer(file);

    for sample in batch {
        let state = sample.state.to_data().convert::<f32>().clone();
        let state = state.as_slice::<f32>().unwrap();
        let next = sample.next_state.to_data().convert::<f32>().clone();
        let next = next.as_slice::<f32>().unwrap();

        writer.serialize(Row {
            action: sample.action,
            reward: sample.reward,
            state_0: state[0],
            state_1: state[1],
            state_2: state[2],
            state_3: state[3],
            state_4: state[4],
            state_5: state[5],
            state_6: state[6],
            state_7: state[7],
            state_8: state[8],
            state_9: state[9],
            state_10: state[10],
            state_11: state[11],
            next_0: next[0],
            next_1: next[1],
            next_2: next[2],
            next_3: next[3],
            next_4: next[4],
            next_5: next[5],
            next_6: next[6],
            next_7: next[7],
            next_8: next[8],
            next_9: next[9],
            next_10: next[10],
            next_11: next[11],
        }).unwrap();
    }

    writer.flush().unwrap();
    println!("✅ Replay 저장 완료: {}", filename);
}
