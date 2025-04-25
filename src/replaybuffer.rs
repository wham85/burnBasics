use crate::replay_log::ReplaySample;
use rand::prelude::IndexedRandom;
use rand::seq::IteratorRandom;
use rand::{seq::SliceRandom, thread_rng};
use std::collections::VecDeque;

pub struct ReplayBuffer {
    buffer: VecDeque<ReplaySample>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, sample: ReplaySample) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }

        self.buffer.push_back(sample);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<ReplaySample> {
        let mut rng = thread_rng();
        self.buffer
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_ready(&self, batch_size: usize) -> bool {
        self.len() >= batch_size
    }
}
