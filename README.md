This repository contains my solution for the [Yandex Cup ML 2024](https://yandex.ru/cup/ml) competition (**17th place on the LB**, **0.575** public nDCG,	**0.563** private nDCG).

The participants were asked to create an algorithm that would find variations and covers of musical works most similar to the original composition.  

As data, the authors of the problem provided time-compressed CQT spectrograms. Each spectrogram represents 60 seconds taken from the central part of a track and has dimension of (84, 50).

## Installation
Run in your terminal:
```
git clone https://github.com/olisvalue/yacup_ml_csi.git
cd yacup_ml_csi
pip install -r requirements.txt
```

## Data and model weights
All data can be downloaded at [link](https://disk.yandex.ru/d/RjMQIusMf6_L4w). You need to put them in the ```/data``` directory.   
Model weights can be loaded [here](https://disk.yandex.ru/d/9txEH19IBe5SzQ).


## Training

To start training, run:
```
python train.py
```
The config used for training: ```/config/config_train.yaml```.   
It is necessary to train the model for at least 20 epochs.   

Pay attention to the following configuration file parameters:   
1. To train only on a train sample, use ``use_value_for_train: False``. Also, change ``enum_classes: 41616`` to ``num_classes: 39535`` in the ``train`` field.
2. If the system has enough RAM, use ``store_data_in_ram: True`` to speed up training.

## Inference
To get the answers of the model in the test sample of the competition, run:
```
python test.py
```
For the inference, the config used: ``/config/config_test.yaml```

Specify the correct path to the checkpoint of the model to be used for the inference in the ``test` field of the configuration file.

After executing the script, the file with the answers will be located in ``/outputs_test``
