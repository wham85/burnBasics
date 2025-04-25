use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::relu, backend::Backend},
};

#[derive(Module, Debug)]
pub struct DqnModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> DqnModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(12, 32).init(device),
            fc2: LinearConfig::new(32, 16).init(device),
            out: LinearConfig::new(16, 3).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.fc1.forward(input));
        let x = relu(self.fc2.forward(x));
        self.out.forward(x)
    }
}
