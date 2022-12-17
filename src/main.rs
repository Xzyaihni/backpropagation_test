#[allow(unused_imports)]
use rand::Rng;

use neural_net::*;

mod neural_net;


fn correct_outputs(inputs: (f64, f64)) -> (f64, f64)
{
    (inputs.1 * 0.3 - 0.1, inputs.0 * 0.5 + 0.1)
}

fn main()
{
    let mut network = NeuralNet::create(&[2, 2, 2], TransferFunction::Sigmoid, 0.001, 0.2);

    let iterations = 10000;

    const PEAKS_AMOUNT: usize = 84;
    let mut peaks = [0.0;PEAKS_AMOUNT];

    let mut rng = rand::thread_rng();
    for i in 0..iterations
    {
        let input = (rng.gen::<f64>(), rng.gen::<f64>());
        let output = correct_outputs(input.clone());

        network.backpropagate(&[TrainSample{
                inputs: vec![input.0, input.1],
                outputs: vec![output.0, output.1]
            }]);

        let e = |v: f64, c: f64| (v-c).abs();

        let out = network.feedforward(&[input.0, input.1]);

        let err = e(out[0], output.0) + e(out[1], output.1);

        let distance = PEAKS_AMOUNT as f64 / iterations as f64;
        let peak = i as f64 * distance;
        peaks[peak as usize] += err/(iterations as f64 / PEAKS_AMOUNT as f64);
    }

    {
        let max_val = 7.0;
        let height = 30;

        for h in (0..height).rev()
        {
            for peak in peaks
            {
                let amount = (max_val/height as f64 * h as f64) - peak;

                if amount < -0.2
                {
                    print!("█");
                } else if amount < -0.1
                {
                    print!("▓");
                } else if amount < 0.0
                {
                    print!("▒");
                } else
                {
                    print!("░");
                }
            }
            println!();
        }
    }

    let test = vec![0.5, 0.7];

    println!("correct: {:?}", correct_outputs((test[0], test[1])));
    println!("network: {:?}", network.feedforward(&test));
}
