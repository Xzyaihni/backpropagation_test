use std::{
    fs::File,
    io
};

use serde::{Serialize, Deserialize};

use rand::Rng;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferFunction
{
    Relu,
    Sigmoid
}

impl TransferFunction
{
    pub fn t_f(&self, n: f64) -> f64
    {
        match self
        {
            TransferFunction::Relu => n.max(0.0),
            TransferFunction::Sigmoid => 0.5 + 0.5 * (0.5*n).tanh()
        }
    }

    //dtf? funy
    pub fn dt_f(&self, n: f64) -> f64
    {
        match self
        {
            TransferFunction::Relu => if n>0.0 {1.0} else {0.0},
            TransferFunction::Sigmoid =>
            {
                0.25 - 0.25 * (0.5*n).tanh().powi(2)
            }
        }
    }
}

pub struct TrainSample
{
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNet
{
    layers: Vec<usize>,
    #[serde(skip)]
    neurons: Vec<Vec<f64>>,

    #[serde(skip)]
    previous_gradient: Vec<Vec<Vec<f64>>>,
    #[serde(skip)]
    gradient_batch: Vec<Vec<Vec<f64>>>,
    weights: Vec<Vec<Vec<f64>>>,

    transfer_function: TransferFunction,

    learning_rate: f64,
    momentum: f64
}

struct DefaultFields
{
    neurons: Vec<Vec<f64>>,
    previous_gradient: Vec<Vec<Vec<f64>>>,
    gradient_batch: Vec<Vec<Vec<f64>>>
}

#[allow(dead_code)]
impl NeuralNet
{
    pub fn create(
        layers: &[usize],
        transfer_function: TransferFunction,
        learning_rate: f64,
        momentum: f64
    ) -> Self
    {
        let DefaultFields{neurons, previous_gradient, gradient_batch} =
            Self::initialize_defaults(layers);

        let mut rng = rand::thread_rng();
        let weights = gradient_batch.iter().map(|layer|
        {
            layer.iter().map(|w|
            {
                w.iter().map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect::<Vec<f64>>()
            }).collect::<Vec<Vec<f64>>>()
        }).collect::<Vec<Vec<Vec<f64>>>>();

        NeuralNet{
            layers: layers.to_vec(),
            neurons,
            previous_gradient, gradient_batch,
            weights,
            transfer_function,
            learning_rate, momentum
        }
    }

    fn initialize_defaults(layers: &[usize]) -> DefaultFields
    {
        assert!(layers.len()>1);

        let neurons = layers.iter().skip(1).map(|layer| vec![0.0; *layer])
            .collect::<Vec<Vec<f64>>>();

        let gradient_batch = (1..layers.len()).map(|layer|
        {
            (0..layers[layer]).map(|_|
            {
                vec![0.0; layers[layer-1]+1]
            }).collect::<Vec<Vec<f64>>>()
        }).collect::<Vec<Vec<Vec<f64>>>>();

        let previous_gradient = gradient_batch.clone();

        DefaultFields{neurons, previous_gradient, gradient_batch}
    }

    pub fn load(filename: &str) -> Result<Self, ciborium::de::Error<io::Error>>
    {
        let mut net = ciborium::de::from_reader::<Self, _>(File::open(filename)
            .map_err(|err| ciborium::de::Error::Io(err))?)?;

        DefaultFields{
            neurons: net.neurons,
            previous_gradient: net.previous_gradient,
            gradient_batch: net.gradient_batch
        } = Self::initialize_defaults(&net.layers);

        Ok(net)
    }

    pub fn save(&self, filename: &str) -> Result<(), ciborium::ser::Error<io::Error>>
    {
        ciborium::ser::into_writer(&self, File::create(filename)
            .map_err(|err| ciborium::ser::Error::Io(err))?)
    }

    pub fn feedforward(&mut self, inputs: &[f64]) -> Vec<f64>
    {
        self.feedforward_inner(inputs);

        self.neurons.last().unwrap().iter().map(|neuron|
        {
            self.transfer_function.t_f(*neuron)
        }).collect::<Vec<f64>>()
    }

    fn feedforward_inner(&mut self, inputs: &[f64])
    {
        unsafe
        {
        for layer in 0..self.layers.len()-1
        {
            for i_neuron in 0..*self.layers.get_unchecked(layer+1)
            {
                let neuron_weights = self.weights.get_unchecked(layer)
                    .get_unchecked(i_neuron);

                let bias = neuron_weights.get_unchecked(*self.layers.get_unchecked(layer));

                self.neurons[layer][i_neuron] =
                    (0..*self.layers.get_unchecked(layer))
                        .map(|i_previous|
                        {
                            let neuron = if layer==0
                                {
                                    *inputs.get_unchecked(i_previous)
                                } else
                                {
                                    self.transfer_function.t_f(*self.neurons
                                        .get_unchecked(layer-1)
                                        .get_unchecked(i_previous))
                                };

                            neuron_weights.get_unchecked(i_previous) * neuron
                        }).sum::<f64>() + bias;
            }
        }
        }
    }

    pub fn backpropagate(&mut self, samples: &[TrainSample])
    {
        for sample in samples
        {
            self.feedforward_inner(&sample.inputs);
            self.backpropagate_inner(&sample.inputs, &sample.outputs);
        }

        unsafe
        {
        for layer in 0..self.layers.len()-1
        {
            for neuron in 0..*self.layers.get_unchecked(layer+1)
            {
                self.gradient_batch.get_unchecked_mut(layer)
                    .get_unchecked_mut(neuron).iter_mut().enumerate()
                    .for_each(|(previous, gradient)|
                    {
                        let previous_gradient = self.previous_gradient.get_unchecked_mut(layer)
                            .get_unchecked_mut(neuron)
                            .get_unchecked_mut(previous);

                        let change = (*gradient + *previous_gradient) / samples.len() as f64;

                        *self.weights.get_unchecked_mut(layer)
                            .get_unchecked_mut(neuron)
                            .get_unchecked_mut(previous) -= change * self.learning_rate;

                        *previous_gradient = *gradient;

                        *gradient = 0.0;
                    });
            }
        }
        }
    }

    fn backpropagate_inner(&mut self, inputs: &[f64], outputs: &[f64])
    {
        //yolo
        unsafe
        {
        for layer in (0..self.layers.len()-1).rev()
        {
            for i_neuron in 0..*self.layers.get_unchecked(layer+1)
            {
                let neuron = self.neurons.get_unchecked(layer).get_unchecked(i_neuron);

                let error = if layer==self.layers.len()-2
                {
                    if cfg!(not(test))
                    {
                        self.transfer_function.t_f(*neuron) - outputs.get_unchecked(i_neuron)
                    } else
                    {
                        1.0
                    }
                } else
                {
                    (0..*self.layers.get_unchecked(layer+2)).map(|i_next|
                    {
                        self.weights.get_unchecked(layer+1)
                            .get_unchecked(i_next)
                            .get_unchecked(i_neuron)
                            * self.neurons.get_unchecked(layer+1).get_unchecked(i_next)
                    }).sum()
                };

                let d_transfer = self.transfer_function.dt_f(*neuron);
                let deriv = error * d_transfer;

                let neuron_gradient = self.gradient_batch
                    .get_unchecked_mut(layer)
                    .get_unchecked_mut(i_neuron);

                for i_previous in 0..*self.layers.get_unchecked(layer)
                {
                    let previous = if layer>0
                    {
                        self.transfer_function.t_f(*self.neurons
                            .get_unchecked(layer-1)
                            .get_unchecked(i_previous))
                    } else
                    {
                        *inputs.get_unchecked(i_previous)
                    };

                    *neuron_gradient.get_unchecked_mut(i_previous) += deriv * previous;
                }

                //bias gradient
                *neuron_gradient.last_mut().unwrap() += deriv;

                //for previous layer's gradient calculation (current gradient)
                *self.neurons.get_unchecked_mut(layer).get_unchecked_mut(i_neuron) = deriv;
            }
        }
        }
    }
}


#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn backprop()
    {
        for _ in 0..10
        {
            backprop_single();
        }
    }

    fn backprop_single()
    {
        let mut rng = rand::thread_rng();
        let layers = [2, rng.gen_range(1..10), rng.gen_range(1..10), rng.gen_range(1..10), 2];

        let mut network = NeuralNet::create(
            &layers,
            TransferFunction::Sigmoid,
            1.0,
            0.0
        );

        let change = 0.2;

        for t_l in 0..layers.len()-1
        {
            for t_n in 0..layers[t_l+1]
            {
                for t_b in 0..(layers[t_l]+1)
                {
                    let test_input = (0..layers[0]).map(|_| rng.gen())
                        .collect::<Vec<f64>>();

                    //doesnt matter
                    let test_output = (0..*layers.last().unwrap()).map(|_| 0.0)
                        .collect::<Vec<f64>>();


                    let normal_weight = network.weights[t_l][t_n][t_b];

                    network.weights[t_l][t_n][t_b] = normal_weight + change;
                    let output = network.feedforward(&test_input);
                    let left = output.into_iter().sum::<f64>();

                    network.weights[t_l][t_n][t_b] = normal_weight - change;
                    let output = network.feedforward(&test_input);
                    let right = output.into_iter().sum::<f64>();

                    network.weights[t_l][t_n][t_b] = normal_weight;

                    network.backpropagate(&[TrainSample{inputs: test_input, outputs: test_output}]);

                    let deriv = network.previous_gradient[t_l][t_n][t_b];
                    let real_deriv = (left - right) / (2.0 * change);

                    print!("(layer: {t_l} neuron: {t_n} previous: {t_b})  ");
                    println!("backprop: {deriv}, derivative: {real_deriv}");
                    println!("diff: {}", deriv-real_deriv);

                    assert!((deriv-real_deriv).abs()<0.001);
                }
            }
        }
    }
}