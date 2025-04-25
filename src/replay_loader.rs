use crate::replay_log::ReplaySample;
use crate::types::B;

use burn::tensor::{Tensor, backend::Backend};
use csv::Reader;
use serde::Deserialize;

#[derive(Deserialize)]
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

pub fn load_replay_csv(filename: &str, device: &<B as Backend>::Device) -> Vec<ReplaySample> {
    let mut rdr = Reader::from_path(filename).unwrap();
    let mut samples = Vec::new();

    for result in rdr.deserialize() {
        let row: Row = result.unwrap();

        let state = Tensor::<B, 2>::from_floats(
            [[
                row.state_0,
                row.state_1,
                row.state_2,
                row.state_3,
                row.state_4,
                row.state_5,
                row.state_6,
                row.state_7,
                row.state_8,
                row.state_9,
                row.state_10,
                row.state_11,
            ]],
            device,
        );

        let next_state = Tensor::<B, 2>::from_floats(
            [[
                row.next_0,
                row.next_1,
                row.next_2,
                row.next_3,
                row.next_4,
                row.next_5,
                row.next_6,
                row.next_7,
                row.next_8,
                row.next_9,
                row.next_10,
                row.next_11,
            ]],
            device,
        );

        samples.push(ReplaySample {
            state,
            action: row.action,
            reward: row.reward,
            next_state,
        });
    }

    samples
}
