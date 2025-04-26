use crate::dqn_model::DqnModel;
use crate::replay_log::ReplaySample;
use crate::types::B;
use burn::tensor::{backend::Backend, Tensor};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, Optimizer, Adam};
use burn::optim::adaptor::OptimizerAdaptor;
use std::fs::File;
use std::path::Path;
use burn::record::{CompactRecorder, Recorder};

pub fn train_from_csv(csv_path: &str, model: &mut DqnModel<B>) {
    println!("📚 학습 시작: {}", csv_path);

    let samples = load_samples_from_csv(csv_path);

    let device = <B as Backend>::Device::default();
    let optimizer: OptimizerAdaptor<Adam, DqnModel<B>, B> = AdamConfig::new()
    .init::<B, DqnModel<B>>()
    .into();
    let loss_fn = MseLoss::new();

    for sample in samples {
        let pred = model.forward(sample.state.clone());
        let next_q = model.forward(sample.next_state.clone());

        let pred_data = pred.to_data().convert::<f32>().clone();
        let pred_data = pred_data.as_slice::<f32>().unwrap();
        let next_data = next_q.to_data().convert::<f32>().clone();
        let next_data = next_data.as_slice::<f32>().unwrap();
        let max_next_q = next_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let target = sample.reward + 0.9 * max_next_q;

        println!(
            "🎯 액션: {}, 예측값: {:.3}, 타겟값: {:.3}",
            sample.action, pred_data[sample.action], target
        );
    }

    println!("✅ 학습 완료");
}

fn load_samples_from_csv(path: &str) -> Vec<ReplaySample> {
    // TODO: CSV 파싱해서 Vec<ReplaySample>로 변환하는 부분
    todo!()
}
