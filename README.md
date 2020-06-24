# Counterfactual Propagation for Semi-SupervisedIndividual Treatment Effect Estimation



Implementation of "Counterfactual Propagation for Semi-SupervisedIndividual Treatment Effect Estimation".

- README.md: this
- requirements.txt: required packages
- src: main source codes



## Dataset

We use datasets (IHDP and News) employed in the previous work. (See https://github.com/clinicalml/cfrnet and authors web page for detail.)

Please follow the description and download the corresponding dataset.

## How to use

Make data directory and put the datasets there.

> mkdir data



> pip install -r requirements.txt
>
> cd src



To run a single experiment, you can run do as follows:

> python train.py --data news -e 2000 -ratio 0.01 -alpha 0 -beta 0 -b 32 -g 0 -s 0



To run grid search or search parameters, you can run as follows:

> python param_search.py

Param_search.py includes parameter dictionaries.

You can search parameters by changing the parameters included in this dictionary.
