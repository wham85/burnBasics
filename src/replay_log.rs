use crate::types::B;
use burn::tensor::{backend::Backend, Tensor};

/// 🧠 상태, 행동, 보상, 다음 상태를 저장하는 구조체
/// DQN에서는 이 경험을 기반으로 학습합니다
#[derive(Clone)]
pub struct ReplaySample {
    /// 현재 상태 (state)
    pub state: Tensor<B, 2>,
    /// 선택한 행동 (0 = Buy, 1 = Sell, 2 = Hold)
    pub action: usize,
    /// 해당 행동을 했을 때의 보상 (reward)
    pub reward: f32,
    /// 행동 이후 도달한 상태 (next_state)
    pub next_state: Tensor<B, 2>,
}
