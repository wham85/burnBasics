use crate::replay_log::ReplaySample;
use burn::tensor::Tensor;
use csv::Writer;
use serde::Serialize;
use std::fs::File;

#[derive(Serialize)]
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

pub fn save_replay_csv(batch: &[ReplaySample], filename: &str) {
    let file = File::create(filename).unwrap();
    let mut writer = Writer::from_writer(file);

    for sample in batch {
        let state_data = sample.state.to_data().convert::<f32>();
        let state = state_data.as_slice::<f32>().unwrap();
        let next_data = sample.next_state.to_data().convert::<f32>();
        let next = next_data.as_slice::<f32>().unwrap();

        writer
            .serialize(Row {
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
            })
            .unwrap();
    }

    writer.flush().unwrap();
    println!("✅ Replay 100개 저장 완료 → {}", filename);
}
