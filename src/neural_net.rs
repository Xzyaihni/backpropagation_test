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
            TransferFunction::Sigmoid => 0.5 + 0.5 * (n/2.0).tanh()
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
                0.25 - 0.25 * (n/2.0).tanh() * (n/2.0).tanh()
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
    gradient_batch: Vec<Vec<Vec<f64>>>,
    weights: Vec<Vec<Vec<f64>>>,

    transfer_function: TransferFunction,

    learning_rate: f64
}

struct DefaultFields
{
    neurons: Vec<Vec<f64>>,
    gradient_batch: Vec<Vec<Vec<f64>>>
}

#[allow(dead_code)]
impl NeuralNet
{
    pub fn create(
        layers: &[usize],
        transfer_function: TransferFunction,
        learning_rate: f64
    ) -> Self
    {
        let DefaultFields{neurons, gradient_batch} = Self::initialize_defaults(layers);

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
            gradient_batch, weights,
            transfer_function, learning_rate
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

        DefaultFields{neurons, gradient_batch}
    }

    pub fn load(filename: &str) -> Result<Self, ciborium::de::Error<io::Error>>
    {
        let mut net = ciborium::de::from_reader::<Self, _>(File::open(filename)
            .map_err(|err| ciborium::de::Error::Io(err))?)?;

        DefaultFields{neurons: net.neurons, gradient_batch: net.gradient_batch} =
            Self::initialize_defaults(&net.layers);

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
            let next_layer_len = self.layers.get_unchecked(layer+1);
            for i_neuron in 0..*next_layer_len
            {
                let neuron_weights = self.weights.get_unchecked(layer)
                    .get_unchecked(i_neuron);

                let bias = neuron_weights.get_unchecked(*next_layer_len);

                self.neurons[layer][i_neuron] =
                    (0..neuron_weights.len()-1)
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
            self.backpropagate_inner(&sample.outputs);
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
                        *self.weights.get_unchecked_mut(layer)
                            .get_unchecked_mut(neuron)
                            .get_unchecked_mut(previous) -=
                            *gradient * self.learning_rate;

                        *gradient = 0.0;
                    });
            }
        }
        }
    }

    fn backpropagate_inner(&mut self, outputs: &[f64])
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
                    self.transfer_function.t_f(*neuron) - outputs.get_unchecked(i_neuron)
                } else
                {
                    (0..*self.layers.get_unchecked(layer+2)).map(|i_next|
                    {
                        self.weights.get_unchecked(layer)
                            .get_unchecked(i_next)
                            .get_unchecked(i_neuron)
                            * self.neurons.get_unchecked(layer+1).get_unchecked(i_next)
                    }).sum()
                };

                let d_transfer = self.transfer_function.dt_f(*neuron);
                let deriv = error * d_transfer;

                *self.neurons.get_unchecked_mut(layer).get_unchecked_mut(i_neuron) = deriv;

                let neuron_gradient = self.gradient_batch
                    .get_unchecked_mut(layer)
                    .get_unchecked_mut(i_neuron);

                (0..neuron_gradient.len()-1).for_each(|i_previous|
                {
                    let gradient = deriv * if layer>0
                    {
                        self.transfer_function.t_f(*self.neurons
                            .get_unchecked(layer-1)
                            .get_unchecked(i_previous))
                    } else
                    {
                        1.0
                    };

                    *neuron_gradient.get_unchecked_mut(i_previous) += gradient;
                });

                //bias gradient
                *neuron_gradient.last_mut().unwrap() = deriv;
            }
        }
        }
    }
}