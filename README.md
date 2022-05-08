# CS6910 ASSIGNMENT 3

## Contents

### Jupyter Notebook files
1. [cs6910_assignment3.ipynb](https://github.com/mak109/cs6910_assignment3/blob/main/cs6910_assignment3.ipynb) : Contains the code from question1 to question 3.
2. [predictions_vanilla](https://github.com/mak109/cs6910_assignment3/tree/main/predictions_vanilla) : Contains the [predictions.csv](https://github.com/mak109/cs6910_assignment3/blob/main/predictions_vanilla/predictions.csv) file which has the top K predictions for each input from the test data (using the best non-attention/vanilla model) written to it.
3. [predictions_attention](https://github.com/mak109/cs6910_assignment3/tree/main/predictions_attention) : Contains the [predictions_attentions.csv](https://github.com/mak109/cs6910_assignment3/blob/main/predictions_attention/predictions_attention.csv)which has the top K predictions for each input from test data (using the best attention based model) written to it.
4. [cs6910_assignment3_attention.ipynb](https://github.com/mak109/cs6910_assignment3/blob/main/cs6910_assignment3_attention.ipynb) : Contains the code for the attention(question 4 to question5).
5. [cs6910_assignment3_gpt.ipynb](https://github.com/mak109/cs6910_assignment3/blob/main/cs6910_assignment3_gpt.ipynb) : Contains code for the question 8.
6. [cs6910_assignment3_viz.ipynb](https://github.com/mak109/cs6910_assignment3/blob/main/cs6910_assignment3_viz.ipynb) : Contains code for visualization(question 6) and also the code for calculating the test and training accuracy.

## Steps to run the code:
To run the Jupyter notebook files use the following steps :
 
### Jupyter Notebook
To run he Jupter Notebook files download the files and run all the cells in sequential order. We recommend to use Google Colab for using the ipynb files so that there is no dependencies error. 

Download all the files and run all the cells sequentially. We recommend to run the files in Google Colab. For the cs6910_assignment3_viz.ipynb file, we recommed to use Kaggle to get the required results because of better GPU in Kaggle as compared to Google Colab.Kaggle provides **Nvidia Tesla P100** and 
also provides better runtime with larger quota.


## Hyperparameters and sweep configs

Following is the hyperparameters for cs6910_assignment3.ipynb
```python
sweep_config_1 = {
    'name': 'RNN',
    'method': 'bayes',                   # Possible search : grid, random, bayes
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate':{
            'values':[1e-3,1e-4,5e-3]
        },
        'epochs':
        {
         'values':[10,25,20]
        },
        'optimizer':{
            'values':['rmsprop','adam','nadam','sgd','momentum']
        },
        'loss_function':{
          'value':'categorical_crossentropy' 
        },
        'input_embedding_size': {
            'values': [64,128, 256,512]
        },
        'num_enc_layers': {
            'values': [1, 2, 3]
        },
        'num_dec_layers': {
            'values': [1, 2, 3]
        },
        'hidden_layer_size': {
            'values': [128,256,512,768]
        },
        'cell_type': {
            'values': ['RNN', 'LSTM', 'GRU']
        },
        'dropout' :{
            'values': [0, 0.25, 0.3,0.4]
        },
        'r_dropout':{
          'values': [0.0,0.20,0.30]  
        },
        'beam_width': {
            'values': [1, 3,5]
        }
    }
}

```

Following is the hyperparameters for cs6910_assignment3_attention.ipynb
```python
sweep_config_1 = {
    'name': 'RNNs2s_attn',
    'method': 'bayes',                   # Possible search : grid, random, bayes
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate':{
            'values':[1e-3,1e-4,5e-3]
        },
        'epochs':
        {
         'values':[10,25,20]
        },
        'optimizer':{
            'values':['rmsprop','adam','nadam','sgd','momentum']
        },
        'loss_function':{
          'value':'categorical_crossentropy' 
        },
        'input_embedding_size': {
            'values': [64,128, 256,512]
        },
        'num_enc_layers': {
            'values': [1, 2, 3]
        },
        'num_dec_layers': {
            'values': [1, 2, 3]
        },
        'hidden_layer_size': {
            'values': [128,256,512,768]
        },
        'cell_type': {
            'values': ['RNN', 'LSTM', 'GRU']
        },
        'dropout' :{
            'values': [0, 0.25, 0.3,0.4]
        },
        'r_dropout':{
          'values': [0.0,0.20,0.30]  
        },
        'beam_width': {
            'values': [1, 3,5]
        }
    }
}
```
Following is the hyperparameters which gives the best word level and character level accuracy for vanilla model is

```python
config = {
    batch_size: 256
    beam_width: 3
    cell_type: 'GRU'
    dropout: 0.3
    epochs: 10
    hidden_layer_size: 256
    input_embedding_size: 256
    learning_rate: 0.001
    loss_function: 'categorical_crossentropy'
    num_dec_layers: 3
    num_enc_layers: 2
    optimizer: 'adam'
    r_dropout: 0.3
    }
```
Following is the hyperparameters which gives the best word level and character level accuracy for attention based model is
```python
config = {
    batch_size: 256
    beam_width: 3
    cell_type: 'GRU'
    dropout: 0.2
    epochs: 20
    hidden_layer_size: 768
    input_embedding_size: 128
    learning_rate: 0.001
    loss_function: 'categorical_crossentropy'
    num_dec_layers: 5
    num_enc_layers: 3
    optimizer: 'adam'
    r_dropout: 0.2
    }
```

