# ELECTRA

This directory contains our implementation of the model from scratch.  

It requires python 3.7+ and transformers 4.1.1. 

Please download and unzip the MuTual dataset and put it in `11747-final-project/MuTual`

Run `train.sh` to train a large ELECTRA model on MuTual. 
Run `test.sh` with the corresponding model paths for testing. 

`python eval_metrics.py [output_dir/output.txt]` will print the model's performance on the official MuTual metrics.  
 