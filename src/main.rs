
mod websocket;

use std::io::{self, Write};
use tokio::sync::mpsc;
use tokio::time::{Duration, timeout};
use websocket::{OrderBookData, TickData, upbit_websocket_handler};


#[tokio::main]
async fn main() {
    let coin = get_coin_symbol();

    let (tick_sender, mut tick_receiver) = mpsc::channel::<TickData>(100);
    let (order_sender, mut order_receiver) = mpsc::channel::<OrderBookData>(100);

    println!("MAIN");
    tokio::spawn(
    upbit_websocket_handler(coin.clone(), tick_sender,order_sender));

    loop {
        if let Ok(Some(order)) = timeout(Duration::from_millis(10), order_receiver.recv()).await {
            println!("{:?}", order);
        }

        if let Ok(Some(tick)) = timeout(Duration::from_millis(10), tick_receiver.recv()).await {
            println!("{:?}", tick);
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