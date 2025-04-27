use crate::dqn_model::DqnModel;
use crate::replay_log::ReplaySample;
use crate::types::B;
use burn::tensor::{backend::Backend, Tensor};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};
use std::fs::File;
use std::error::Error;
use csv::Reader;
use std::io::BufReader;

/// CSV 파일을 불러와서 ReplaySample 리스트로 변환하는 함수
pub fn load_samples_from_csv(filename: &str) -> Result<Vec<ReplaySample>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let mut reader = Reader::from_reader(BufReader::new(file));

    let device = <B as Backend>::Device::default();
    let mut samples = Vec::new();

    for result in reader.records() {
        let record = result?;

        let action: usize = record.get(0).unwrap().parse()?;
        let reward: f32 = record.get(1).unwrap().parse()?;

        let mut state_vec = Vec::new();
        for i in 2..14 {
            let value: f32 = record.get(i).unwrap().parse()?;
            state_vec.push(value);
        }
        let state = Tensor::<B, 2>::from_floats(state_vec.as_slice(), &device)
            .reshape([1, state_vec.len()]);

        let mut next_state_vec = Vec::new();
        for i in 14..26 {
            let value: f32 = record.get(i).unwrap().parse()?;
            next_state_vec.push(value);
        }
        let next_state = Tensor::<B, 2>::from_floats(next_state_vec.as_slice(), &device)
            .reshape([1, next_state_vec.len()]);

        samples.push(ReplaySample {
            state,
            action,
            reward,
            next_state,
        });
    }

    Ok(samples)
}

/// CSV 파일을 읽어와서 모델을 학습시키는 함수
pub fn train_from_csv(csv_path: &str, model: &mut DqnModel<B>) -> Result<(), Box<dyn Error>> {
    println!("📚 학습 시작: {}", csv_path);

    let samples = load_samples_from_csv(csv_path)?;
    let device = <B as Backend>::Device::default();
    let mut optimizer = AdamConfig::new().init::<B, DqnModel<B>>();
    let loss_fn = MseLoss::new();

    let learning_rate = 0.001;

    for sample in samples.iter() {
        let pred = model.forward(sample.state.clone());
        let next_q = model.forward(sample.next_state.clone());

        let pred_data = pred.to_data().convert::<f32>();
        let pred_data = pred_data.as_slice::<f32>().unwrap();
        let next_data = next_q.to_data().convert::<f32>();
        let next_data = next_data.as_slice::<f32>().unwrap();

        let max_next_q = next_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let target = sample.reward + 0.9 * max_next_q;

        println!(
            "🎯 액션: {}, 예측값: {:.3}, 타겟값: {:.3}",
            sample.action, pred_data[sample.action], target
        );

        let mut target_vec = pred_data.to_vec();
        target_vec[sample.action] = target;

        let target_tensor = Tensor::<B, 2>::from_floats(target_vec.as_slice(), &device)
            .reshape([1, target_vec.len()]);
        let pred_tensor = pred.reshape([1, pred.shape().dims::<2>()[0]]);

        let loss = loss_fn.forward(pred_tensor, target_tensor, Reduction::Mean);

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, model);

        *model = optimizer.step(learning_rate, model.clone(), grads_params);
    }

    println!("✅ 학습 완료");
    Ok(())
}
