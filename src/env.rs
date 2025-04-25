use crate::analyzer::MarketFeatures;
use crate::websocket::TickData;
use burn::tensor::{Tensor, backend::Backend};

pub struct Env<B: Backend> {
    pub device: B::Device,
    pub features: [f32; 12], // 12개 피처 저장
    pub is_holding: bool,    // 현재 포지션 보유 여부
    pub entry_price: f32,    // 매수 진입 가격
}

impl<B: Backend> Env<B> {
    /// 🔧 초기화
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            features: [0.0; 12],
            is_holding: false,
            entry_price: 0.0,
        }
    }

    /// ✅ 분석기 결과를 기반으로 상태 업데이트
    pub fn update(&mut self, f: MarketFeatures) {
        self.features = [
            f.avg_price,
            f.price_delta,
            f.volume_sum,
            f.volatility,
            f.imbalance,
            f.spread,
            f.ask1_price,
            f.bid1_price,
            f.ask_depth_ratio,
            f.bid_depth_ratio,
            f.tick_speed,
            f.last_tick_size,
        ];
    }

    /// 🧠 현재 상태를 Tensor로 반환
    pub fn observe(&self) -> Tensor<B, 2> {
        Tensor::from_floats([self.features], &self.device)
    }

    /// ⚔️ 에이전트의 행동에 따라 포지션/보상 계산
    /// action: 0 = Buy, 1 = Sell, 2 = Hold
    pub fn step(&mut self, action: usize, tick: TickData) -> (Tensor<B, 2>, f32) {
        let current_price = tick.price;
        let mut reward = 0.0;

        match action {
            0 => {
                // Buy
                if !self.is_holding {
                    self.is_holding = true;
                    self.entry_price = current_price;
                    reward = 0.0; // 진입은 보상 없음
                } else {
                    reward = -0.01; // 중복 진입 패널티
                }
            }
            1 => {
                // Sell
                if self.is_holding {
                    let profit = current_price - self.entry_price;
                    reward = profit / self.entry_price; // 수익률 기반 보상
                    self.is_holding = false;
                    self.entry_price = 0.0;
                } else {
                    reward = -0.01; // 없는 포지션에서 매도 패널티
                }
            }
            _ => {
                // Hold
                reward = 0.0;
            }
        }

        // 상태 일부 갱신 (Tick 반영)
        self.features[0] = tick.price; // avg_price
        self.features[2] = tick.volume; // volume_sum
        self.features[11] = tick.volume; // last_tick_size

        let next_state = self.observe();
        (next_state, reward)
    }
}
