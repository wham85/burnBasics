use crate::websocket::{OrderBookData, OrderBookUnit, TickData};
use std::collections::VecDeque;

/// 📦 실시간 Tick / OrderBook 데이터를 저장하는 순환 버퍼 구조
pub struct MarketStorage {
    pub ticks: VecDeque<TickData>,
    pub orderbooks: VecDeque<OrderBookData>,
    pub capacity: usize,
}

impl MarketStorage {
    /// 🔧 새로운 저장소 생성
    pub fn new(capacity: usize) -> Self {
        Self {
            ticks: VecDeque::with_capacity(capacity),
            orderbooks: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// ✅ Tick 데이터 추가
    pub fn push_tick(&mut self, tick: TickData) {
        if self.ticks.len() == self.capacity {
            self.ticks.pop_front(); // 오래된 것 제거
        }
        self.ticks.push_back(tick);
    }

    /// ✅ OrderBook 데이터 추가
    pub fn push_orderbook(&mut self, ob: OrderBookData) {
        if self.orderbooks.len() == self.capacity {
            self.orderbooks.pop_front(); // 오래된 것 제거
        }
        self.orderbooks.push_back(ob);
    }
}

/// 📈 분석된 피처들을 담는 구조체 (총 12개)
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    pub avg_price: f32,
    pub price_delta: f32,
    pub volume_sum: f32,
    pub volatility: f32,
    pub imbalance: f32,
    pub spread: f32,
    pub ask1_price: f32,
    pub bid1_price: f32,
    pub ask_depth_ratio: f32,
    pub bid_depth_ratio: f32,
    pub tick_speed: f32,
    pub last_tick_size: f32,
}

/// 🔍 저장된 데이터를 기반으로 피처를 계산하는 분석 함수
pub fn analyze(storage: &MarketStorage) -> Option<MarketFeatures> {
    // 최소 2개의 틱, 1개의 오더북이 있어야 분석 가능
    if storage.ticks.len() < 2 || storage.orderbooks.is_empty() {
        return None;
    }

    let ticks = &storage.ticks;
    let orderbook = storage.orderbooks.back().unwrap();
    let units = &orderbook.order_units;

    // 1️⃣ 평균 체결가
    let avg_price: f32 = ticks.iter().map(|t| t.price).sum::<f32>() / ticks.len() as f32;

    // 2️⃣ 체결가 변화량 (가장 오래된 것과 최신 것의 차이)
    let price_delta = ticks.back().unwrap().price - ticks.front().unwrap().price;

    // 3️⃣ 총 체결량
    let volume_sum: f32 = ticks.iter().map(|t| t.volume).sum();

    // 4️⃣ 체결가 변동성 (표준편차)
    let mean = avg_price;
    let variance = ticks.iter().map(|t| (t.price - mean).powi(2)).sum::<f32>() / ticks.len() as f32;
    let volatility = variance.sqrt();

    // 5️⃣ 최우선 매도/매수호가 (체결 예상가)
    let ask1_price = units.first().map(|u| u.ask_price).unwrap_or(0.0);
    let bid1_price = units.first().map(|u| u.bid_price).unwrap_or(0.0);

    // 6️⃣ 스프레드 (ask1 - bid1)
    let spread = ask1_price - bid1_price;

    // 7️⃣ 전체 호가 잔량 합
    let ask_sum: f32 = units.iter().map(|u| u.ask_size).sum();
    let bid_sum: f32 = units.iter().map(|u| u.bid_size).sum();

    // 8️⃣ 호가 잔량 불균형
    let imbalance = if ask_sum + bid_sum > 0.0 {
        (bid_sum - ask_sum) / (bid_sum + ask_sum)
    } else {
        0.0
    };

    // 9️⃣ 상위 5호가의 집중도 (depth ratio)
    let ask_top5: f32 = units.iter().take(5).map(|u| u.ask_size).sum();
    let bid_top5: f32 = units.iter().take(5).map(|u| u.bid_size).sum();

    let ask_depth_ratio = if ask_sum > 0.0 {
        ask_top5 / ask_sum
    } else {
        0.0
    };
    let bid_depth_ratio = if bid_sum > 0.0 {
        bid_top5 / bid_sum
    } else {
        0.0
    };

    // 🔟 체결 속도 (틱 간 평균 시간 차이)
    let mut tick_speed = 0.0;
    if ticks.len() >= 2 {
        let mut time_diffs = vec![];
        for w in ticks.iter().collect::<Vec<_>>().windows(2) {
            let diff = w[1].timestamp as i64 - w[0].timestamp as i64;
            time_diffs.push(diff as f32);
        }
        tick_speed = time_diffs.iter().sum::<f32>() / time_diffs.len() as f32;
    }

    // 1️⃣1️⃣ 마지막 체결량 (시장 반응 강도)
    let last_tick_size = ticks.back().map(|t| t.volume).unwrap_or(0.0);

    Some(MarketFeatures {
        avg_price,
        price_delta,
        volume_sum,
        volatility,
        imbalance,
        spread,
        ask1_price,
        bid1_price,
        ask_depth_ratio,
        bid_depth_ratio,
        tick_speed,
        last_tick_size,
    })
}
