# Can Test-time Computation Mitigate Memorization Bias in Neural Symbolic Regression?
[arXiv](https://arxiv.org/abs/2505.22081)

## Installation

To run the code, first create a conda environment and install the dependencies by running the following command.

```
conda create --name mem_bias_nsr
conda activate mem_bias_nsr
pip install -r requirements.txt
```

## Downloading Models

We used 5 models in our experiments, which can be downloaded from the links below.
* A NeSymReS model trained for our controlled setting (used to test on the *not_included* and *baseline* datasets): [here](https://drive.google.com/file/d/1ulWg_UFANGnQWQ3sWSKvqzqCRXdE-ZMy/view?usp=drive_link)
* A NSR-gvs model trained for our controlled setting (used to test on the *not_included* dataset): [here](https://drive.google.com/file/d/1A680DJXatd7Z3X5HC1c323ljTbw1nW2O/view?usp=drive_link)
* A NSRwH model finetuned for our controlled setting (used to test on the *not_included* dataset): [here](https://drive.google.com/file/d/1JsEpr3_pndOlKbt2JooD7RjdkDIHkhuG/view?usp=drive_link)
* A NeSymReS model trained for a more practical setting (used to test on the *ai_feynman* and *only_five_variables_nc* datasets): [here](https://drive.google.com/file/d/1QP0fw99hx663hXRJ9uBFb9BgMx-hAT2h/view?usp=drive_link)
* A NSR-gvs model trained for a more practical setting (used to test on the *ai_feynman* and *only_five_variables_nc* datasets): [here](https://drive.google.com/file/d/1Fs9_nLRXHoHCsKLhGFsQi0fBfj34Pkhf/view?usp=drive_link)

## Test Data Generation

In order to generate test datasets such as *ai_feynman* and *only_five_variables_nc*, please execute the following code.
```
scripts/data_creation/convert_csv_to_dataload_format.py 
```
This script will create a new folder called "benchmark" inside the data folder. Inside this folder, it will create a folder for each benchmark set.
As for the *not_included* dataset and *baseline* dataset, they are already generated as an executable format.

## Testing

Executing exp.py will make a .json file containing the results (the models prediction, R2 value, etc.) under the experiments directory.

To discern whether the output expression is included in the training dataset, please use the `is_expression_in_train_data()` function in memorization_bias_analysis.py.

We provide example scripts to reproduce results. Please refer to the files in experiments/example_shellscripts.


## Training 
Please refer to the explanation for [NSRwH](https://github.com/SymposiumOrganization/ControllableNeuralSymbolicRegression) for the generation of training data.

For training models, we provide example scripts to train the NeSymReS model, NSRwh model, and the NSR-gvs model.

## Citation
```bibtex

@article{sato2025can,
  title={Can Test-time Computation Mitigate Memorization Bias in Neural Symbolic Regression?},
  author={Sato, Shun and Sato, Issei},
  journal={arXiv preprint arXiv:2505.22081},
  year={2025}
}
```
