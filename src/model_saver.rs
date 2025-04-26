// src/model_saver.rs

use crate::dqn_model::DqnModel;
use crate::types::B;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use std::path::Path;
use burn::module::Module;

// 폴더 없이 복사하는 수 있게 model_path는 Path 파라미터로 만들어줌.

pub fn save_model(model: &DqnModel<B>, model_path: &str) {
    let recorder = CompactRecorder::new();
    model.clone()
        .save_file(model_path, &recorder)
        .expect("모델 저장 실패");
}

pub fn load_model(model_path: &str, device: &<B as burn::tensor::backend::Backend>::Device) -> DqnModel<B> {
    let recorder = CompactRecorder::new();

    // 작성된 Record를 로드해서 다시 메뉴 플레이스로 사용
    let record = recorder
        .load(Path::new(model_path).to_path_buf(), device)
        .expect("모델 로드 실패");

    DqnModel::new(device).load_record(record)
}
