## The code for DIFF blocks for classification

This repository provides the Pytorch implementation for the paper: Deep Integrated Fusion of Local and Global Features for Cervical Cell Classification.

### Requirements

The source code was running with the following environment.

* Python3.8
* Pytorch1.10.0
* torchvision
* torchaudio
* matplotlib
* time

### Data preparation

Please download  the following datasets.

* SIPaKMeD (https://www.cs.uoi.gr/~marina/sipakmed.html)
* CRIC (https://database.cric.com.br/)
* Herlev (https://mde-lab.aegean.gr/index.php/downloads/)
* BCCD (https://www.kaggle.com/paultimothymooney/blood-cells)

### How to train the model

Please run as the follows.

#### model.py

The proposed model with DIFF blocks.

#### utils.py

The dataset settings.

#### main.py

The training for the proposed model.

