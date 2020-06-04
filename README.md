# MalRNN
An RNN-based framework to evade deep-learning based malware detectors in black-box mode. E.g., MalConv and Non-Negative MalConv.
[PyTorch implementation of char-rnn](https://github.com/spro/char-rnn.pytorch) is used for reference.
## Requirement
This code is implemented in Python and PyTorch. All of the required libraries could be installed by pip or conda. 
## Usage
### Data
Benign files is given in ../neg. The binary files of objective malware should be in one folder. 
### Training
Before training MalRNN, only rnn_train.py file needs to be changed. In rnn_train.py, change tag_name to any name you want for different purposes, and also change malware_path and save_path to the correct path where you want the model to read and save.
