use std::{
    fs::File,
    io
};

use serde::{Serialize, Deserialize};

use rand::Rng;


pub struct TrainSample
{
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNet
{
    #[serde(skip)]
    neurons: Vec<Vec<f64>>,

    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<f64>,
    learning_rate: f64
}

#[allow(dead_code)]
impl NeuralNet
{
    pub fn create(layers: &[usize], learning_rate: f64) -> Self
    {
        let mut neurons = Vec::new();
        for &layer in layers
        {
            neurons.push(vec![0.0;layer]);
        }

        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();

        for layer in 0..layers.len().checked_sub(1).expect("layers mustn't be empty")
        {
            let mut n_weights = Vec::new();
            for _ in 0..layers[layer+1]
            {
                let mut weights = Vec::new();
                for _ in 0..layers[layer]
                {
                    weights.push(rng.gen());
                }

                n_weights.push(weights);
            }

            weights.push(n_weights);
        }

        let biases = neurons.iter().skip(1).map(|_|
        {
            rng.gen::<f64>()*2.0-1.0
        }).collect::<Vec<f64>>();

        NeuralNet{neurons, weights, biases, learning_rate}
    }

    pub fn load(filename: &str) -> Result<Self, ciborium::de::Error<io::Error>>
    {
        ciborium::de::from_reader::<Self, _>(File::open(filename)
            .map_err(|err| ciborium::de::Error::Io(err))?)
    }

    pub fn save(&self, filename: &str) -> Result<(), ciborium::ser::Error<io::Error>>
    {
        ciborium::ser::into_writer(&self, File::create(filename)
            .map_err(|err| ciborium::ser::Error::Io(err))?)
    }

    pub fn backpropagate(&mut self, sample: TrainSample)
    {
        assert!(self.neurons.len()>1);

        let TrainSample{inputs, outputs: correct_outputs} = sample;
        self.feedforward(inputs);

        let outputs = self.derive_outputs(correct_outputs);
        self.derive_hidden(outputs);
    }

    fn derive_outputs(&mut self, correct_outputs: Vec<f64>) -> Vec<f64>
    {
        let last_layer = self.neurons.len()-1;
        let before_last = last_layer-1;

        let mut out_derivatives = vec![0.0; correct_outputs.len()];

        self.neurons[last_layer].iter().enumerate()
            .for_each(|(i_current, neuron)|
            {
                if *neuron > 0.0
                {
                    self.neurons[before_last].iter().enumerate()
                        .for_each(|(i_previous, previous_neuron)|
                        {
                            let error = neuron.max(0.0) - correct_outputs[i_current];
                            let deriv = error * previous_neuron.max(0.0);

                            out_derivatives[i_current] += error;
                            self.biases[before_last] -= error * self.learning_rate;
                            self.weights[before_last][i_current][i_previous] -=
                                deriv * self.learning_rate;
                        });
                }
            });

        out_derivatives
    }

    fn derive_hidden(&mut self, mut next_derivatives: Vec<f64>)
    {
        let mut neurons = self.neurons.iter().enumerate();
        neurons.next();

        neurons.rev().skip(1).for_each(|(layer, neuron_layer)|
        {
            let mut new_derivatives = vec![0.0; neuron_layer.len()];
            neuron_layer.iter().enumerate()
                .for_each(|(i_current, neuron)|
                {
                    if *neuron > 0.0
                    {
                        self.neurons[layer-1].iter().enumerate()
                            .for_each(|(i_previous, previous_neuron)|
                            {
                        let p_deriv = self.neurons[layer+1].iter().enumerate()
                            .map(|(i_next, _)|
                            {
                                self.weights[layer][i_next][i_current] * next_derivatives[i_next]
                            }).sum::<f64>();

                        new_derivatives[i_current] += p_deriv;

                        let deriv = p_deriv * *previous_neuron;

                        self.biases[layer-1] -= p_deriv * self.learning_rate;
                        self.weights[layer-1][i_current][i_previous] -=
                            deriv * self.learning_rate;
                            });
                    }
                });

            next_derivatives = new_derivatives;
        });
    }

    pub fn feedforward<'a>(&'a mut self, inputs: Vec<f64>) -> &'a Vec<f64>
    {
        self.neurons[0] = inputs;

        for layer in 0..self.neurons.len()-1
        {
            for neuron in 0..self.neurons[layer+1].len()
            {
                self.neurons[layer+1][neuron] = self.neurons[layer].iter()
                    .zip(self.weights[layer][neuron].iter()).map(|(previous, weight)|
                    {
                        previous.max(0.0) * *weight
                    }).sum::<f64>() + self.biases[layer];
            }
        }

        self.neurons.last().unwrap()
    }
}