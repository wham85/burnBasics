use crate::types::B;
use burn::tensor::{Tensor, backend::Backend};
use rand::Rng;

pub struct Agent {
    pub epsilon: f32,
    rng: rand::rngs::ThreadRng,
}

impl Agent {
    pub fn new(epsilon: f32) -> Self {
        Self {
            epsilon,
            rng: rand::thread_rng(),
        }
    }

    //현재 Q값을 바탕으로 행동을 선택(e-greedy)
    pub fn select_action(&mut self, q_array: &[f32]) -> usize {
        //확률적으로 무작위 행동 선택(탐험)
        if self.rng.r#gen::<f32>() < self.epsilon {
            return self.rng.gen_range(0..3);
        }

        q_array
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(2)
    }

    //epsilon 값을 점점 줄여서 탐색을 줄이고 학습을 높임.
    pub fn decay_epsilon(&mut self, decay_rate: f32, min_epsilon: f32) {
        //줄였을 때 더 크면 유지, 더 작아지면 최소값으로 고정
        self.epsilon = (self.epsilon * decay_rate).max(min_epsilon);
    }
}
