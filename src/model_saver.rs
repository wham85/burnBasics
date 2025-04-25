use crate::dqn_model::DqnModel;
use crate::types::B;
use bincode;
use burn::module::Module;
use burn::tensor::backend::Backend;
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// 모델 저장: model → record → bincode 파일
pub fn save_model(model: &DqnModel<B>, path: &str)
where
    B: Backend,
{
    let record = model.clone().into_record();
    let writer = BufWriter::new(File::create(path).unwrap());
    bincode::serialize_into(writer, &record).unwrap();
    println!("💾 모델 저장 완료 → {}", path);
}

/// 모델 불러오기: 파일 → record → model
pub fn load_model(path: &str, device: &<B as Backend>::Device) -> DqnModel<B> {
    let reader = BufReader::new(File::open(path).unwrap());
    let record: <DqnModel<B> as Module<B>>::Record = bincode::deserialize_from(reader).unwrap();
    DqnModel::from_record(record).to_device(device)
}
