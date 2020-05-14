# Factored Machine Translation in Joeynmt

Based on: https://github.com/joeynmt/joeynmt
Factor implementation (except model.py): https://github.com/bricksdont/joeynmt

# Changes made

./configs/

rnn_wmt16_factors_concatenate_deen.yaml & rnn_wmt16_factors_add_deen.yaml

    training: 
        model_dir: <- set to respective directory
    
    model:
        encoder: 
            factor_embeddings:
                embedding_dim : <- set
            factor_combine: <- set
            embeddings:
                embedding_dim : <- set
        decoder:
            embeddings:
                embedding_dim : <- set
                
./joeynmt/
model.py 

    class Model
        
        Added optional factor embeddings and vocabulary to init().
        
        Added optional factors and factor lengths to forward().
        
        Added optional factors and factor lengths to encode().
            Add or concatenate factor and source embeddings when calling self.encoder().
        
        Added optional inclusion of factors in loss calculation in get_loss_for_batch().
        
        Added optional inclusion of factors in run_batch().
        
build_model() 
Added optional factor vocabulary.

    Added optional factor embeddings with padding.
    
    Added embedding dimension check.
    
    Added encoder variants for adding and concatenating of factor & source embeddings
    
# Findings

Baseline:    Bleu = 8.6 | BP 0.838

Concatenate: Bleu = 0.7 | BP 1.0

Add :        Bleu = 0.9 | BP 1.0
        
Interestingly, the factored models give longer translations, resulting in no brevity penalty.
From these initial results, it seems that the required training time is increased massively.
This would need to be re-run on more epochs.
        
# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place.

    git clone https://github.com/[yourusername]/joeynmt
    
Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh
    source venvs/torch3/bin/activate

Then install packages and clone all the required code:
    
    ./scripts/download_install_packages.sh

This will install JoeyNMT and all its dependencies.
Download prepared data for training, vali-dation and testing:
 
    ./scripts/download_preprocessed_data.sh
    
Training the models:
Before  you  run  the  scripts to train the models:  if  you  have  a  GPU,  edit  the JoeyNMT config to enable usecuda.
If you have a different number of cores on CPU, change the scripts: train_add.sh & train_concatenate.sh

Variant a) add factor embeddings to source embeddings:
    
    .scripts/train_add.sh
    
Variant b) concatenate factor embeddings to source embeddings:

    .scripts/train_concatenate.sh
    
Before  you  run  the  scripts to evaluate the models:  if  you  have  a  GPU,  edit  the JoeyNMT config to enable usecuda.
If you have a different number of cores on CPU, change the scripts: evaluate_add.sh & evaluate_concatenate.sh

To compute the test Bleu score:

Variant a):

    ./scripts/evaluate_add.sh
    
Variant b):

    ./scripts/evaluate_concatenate.sh
    
