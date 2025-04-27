use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio::sync::mpsc::Sender;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

// === Tick & OrderBook 구조체 ===

#[derive(Debug, Clone)]
pub struct TickData {
    pub price: f32,
    pub volume: f32,
    pub side: String,   // "ASK" or "BID"
    pub timestamp: u64, // Unix time in milliseconds
}

#[derive(Debug, Clone)]
pub struct OrderBookUnit {
    pub ask_price: f32,
    pub ask_size: f32,
    pub bid_price: f32,
    pub bid_size: f32,
}

#[derive(Debug, Clone)]
pub struct OrderBookData {
    pub timestamp: u64,
    pub order_units: Vec<OrderBookUnit>,
}

pub async fn upbit_websocket_handler(
    coin_code: String,
    tick_sender: Sender<TickData>,
    order_sender: Sender<OrderBookData>,
) {
    println!("first");
    let url = "wss://api.upbit.com/websocket/v1";
    let (ws_stream, _) = connect_async(url).await.expect("[WebSocket] 연결 실패");

    println!("last");
    println!("[WebSocket] 연결 성공: {}", coin_code);
    let (mut write, mut read) = ws_stream.split();

    // 📩 구독 메시지 전송
    let subscribe_msg = json!([
        { "ticket": "test" },
        { "type": "trade", "codes": [coin_code] },
        { "type": "orderbook", "codes": [coin_code] }
    ]);
    let msg = Message::Text(subscribe_msg.to_string().into());
    write.send(msg).await.unwrap();

    // 📥 메시지 수신 루프
    while let Some(Ok(msg)) = read.next().await {
        if let Message::Binary(bin) = msg {
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bin) {
                if let Some(data_type) = value.get("type").and_then(|v| v.as_str()) {
                    match data_type {
                        "trade" => {
                            println!("{:?}", value);
                            if let (Some(price), Some(volume), Some(side), Some(timestamp)) = (
                                value.get("trade_price"),
                                value.get("trade_volume"),
                                value.get("ask_bid"),
                                value.get("timestamp"),
                            ) {
                                let tick = TickData {
                                    price: price.as_f64().unwrap() as f32,
                                    volume: volume.as_f64().unwrap() as f32,
                                    side: side.as_str().unwrap().to_string(),
                                    timestamp: timestamp.as_u64().unwrap(),
                                };
                                let _ = tick_sender.send(tick).await;
                            }
                        }
                        "orderbook" => {
                            if let (Some(orderbook_units), Some(timestamp)) =
                                (value.get("orderbook_units"), value.get("timestamp"))
                            {
                                let units = orderbook_units
                                    .as_array()
                                    .unwrap()
                                    .iter()
                                    .map(|unit| OrderBookUnit {
                                        ask_price: unit["ask_price"].as_f64().unwrap() as f32,
                                        ask_size: unit["ask_size"].as_f64().unwrap() as f32,
                                        bid_price: unit["bid_price"].as_f64().unwrap() as f32,
                                        bid_size: unit["bid_size"].as_f64().unwrap() as f32,
                                    })
                                    .collect();

                                let order_data = OrderBookData {
                                    timestamp: timestamp.as_u64().unwrap(),
                                    order_units: units,
                                };
                                let _ = order_sender.send(order_data).await;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
