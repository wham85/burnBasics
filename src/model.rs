use crate::types::B;

use burn::backend::{ndarray::NdArray, Autodiff};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;
use burn::prelude::Module;
use burn::tensor::activation::relu;
use burn::tensor::Tensor;

#[derive(Module, Debug, Clone)]
pub struct CandyModel {
    layer: Linear<B>,
}

impl CandyModel {
    pub fn new(device: &<B as Backend>::Device) -> Self {
        Self {
            layer: LinearConfig::new(4, 3).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        relu(self.layer.forward(input))
    }
}
