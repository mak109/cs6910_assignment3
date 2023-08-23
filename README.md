# English to Bengali Transliteration

Developed a sequence to sequence model using LSTM,GRU,RNN that converts a given romanized english word or translit-
erate to the native word ( bengali in my case).

Assignment 3 of the **CS6910: Fundamentals of Deep Learning Course by Dr. Mitesh Khapra, CSE, IITM**

Project Contributors
- Mayukh Das(CS21S064)
- Amartya Basu(CS21S063)

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
## Link for Wandb report
[CS6910 Assignmnet 3](https://wandb.ai/dlstack/cs6910_assignment_3/reports/CS6910-Assignment-3--VmlldzoxOTY3NDg4?accessToken=p09lbxbav5s5cwkvcwx1rr3yfvh07axj2xog9bzb6blsqdtei6zwydahr20smx4x)

## References

1) [Deep Learning lectures by Professor Mitesh Khapra, IIT Madras.](https://youtube.com/playlist?list=PLEAYkSg4uSQ1r-2XrJ_GBzzS6I-f8yfRU)
2) [https://www.tensorflow.org](https://www.tensorflow.org)
3) [https://towardsdatascience.com/how-to-visually-explain-any-cnn-based-models-80e0975ce57](https://towardsdatascience.com/how-to-visually-explain-any-cnn-based-models-80e0975ce57)
4) [https://huggingface.co](https://huggingface.co)
5) [https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a](https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a)
6) [https://distill.pub/2019/memorization-in-rnns/#appendix-autocomplete](https://distill.pub/2019/memorization-in-rnns/#appendix-autocomplete)
7) [https://towardsdatascience.com/visualising-lstm-activations-in-keras-b50206da96ff](https://towardsdatascience.com/visualising-lstm-activations-in-keras-b50206da96ff)

## License

The code in this project is made by Mayukh and Amartya.
