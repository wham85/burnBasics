use crate::dqn_model::DqnModel;
use crate::replay_loader::load_replay_csv;
use crate::replay_log::ReplaySample;
use crate::train::train_step;
use crate::types::B;

use burn::optim::{AdamConfig, Optimizer};
use burn::tensor::backend::Backend;

pub fn run_training(csv_path: &str, epochs: usize, batch_size: usize) {
    let device = <B as Backend>::Device::default();

    // 모델 & 옵티마이저 초기화
    let mut model = DqnModel::new(&device);
    let mut optimizer = AdamConfig::new().init::<B, DqnModel>().into();

    // CSV에서 학습 샘플 로드
    let dataset: Vec<ReplaySample> = load_replay_csv(csv_path, &device);

    if dataset.len() < batch_size {
        println!(
            "❗ 데이터가 부족합니다. ({} < {})",
            dataset.len(),
            batch_size
        );
        return;
    }

    println!(
        "🔧 학습 시작: 총 {} 에포크, 배치 크기 {}",
        epochs, batch_size
    );

    for epoch in 1..=epochs {
        // 무작위 셔플 + 배치 추출
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let batch: Vec<ReplaySample> = dataset
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect();

        let (new_model, new_optimizer, loss) = train_step(model, optimizer, batch);
        model = new_model;
        optimizer = new_optimizer;

        println!("📚 Epoch {:>3} | Loss: {:.6}", epoch, loss);
    }

    println!("✅ 학습 완료!");
}
