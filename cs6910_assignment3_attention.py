
# **Installing dependencies**
# !pip install wget
# !pip install --upgrade wandb

# **Importing necessary libraries,packages,etc.**


import wget
import os
import tarfile

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from keras.layers import SimpleRNN,GRU,LSTM,Embedding,Input,Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.metrics import categorical_crossentropy,sparse_categorical_crossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy,CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
#Wandb import and authentication
import wandb
from wandb.keras import WandbCallback
wandb.login(key='b44266d937596fcef83bedbe7330d6cee108a277')

#Dataset downloading and extracting
filename = 'dakshina_dataset_v1.0'
url = 'https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar'
if not os.path.exists(filename+'.tar') and not os.path.exists(filename):
    filename_tar = wget.download(url)
    file = tarfile.open(filename_tar)
    print('\nExtracting files ....')
    file.extractall()
    file.close()
    print('Done')
    os.remove(filename_tar)
elif not os.path.exists(filename):
    filename_tar = filename + '.tar'
    file = tarfile.open(filename_tar)
    print('\nExtracting files ....')
    file.extractall()
    file.close()
    print('Done')
    os.remove(filename_tar)

#Paths
lang = 'bn'
train_path =  filename+f"/{lang}/lexicons/{lang}.translit.sampled.train.tsv"
val_path = filename+f"/{lang}/lexicons/{lang}.translit.sampled.dev.tsv"
test_path = filename+f"/{lang}/lexicons/{lang}.translit.sampled.test.tsv"

def read_data(path):
    df = pd.read_csv(path,header=None,sep='\t')
    df.fillna("NaN",inplace=True)
    input_texts,target_texts = df[1].to_list(),df[0].to_list()
    return input_texts,target_texts

def parse_text(texts):
    characters = set()
    for text in texts:
        for c in text:
            if c not in characters:
                characters.add(c)
    characters.add(' ')
    return sorted(list(characters))

def start_end_pad(texts):
    for i in range(len(texts)):
        texts[i] = "\t" + texts[i] + "\n"
    return texts

#Train test val dataset import raw
train_input_texts,train_target_texts = read_data(train_path)
val_input_texts,val_target_texts = read_data(val_path)
test_input_texts,test_target_texts = read_data(test_path)

#Padding at beginning and end with '\t' and '\n' respectively
train_target_texts = start_end_pad(train_target_texts)
val_target_texts = start_end_pad(val_target_texts)
test_target_texts = start_end_pad(test_target_texts)

import argparse
#Default Configuration for training
config_ = {
    "learning_rate": 1e-3,                                      # Learning rate in gradient descent
    "epochs": 10,                                               # Number of epochs to train the model   
    "optimizer": 'adam',                                        # Gradient descent algorithm used for the parameter updation
    "batch_size": 64,                                           # Batch size used for the optimizer
    "loss_function": 'categorical_crossentropy',                # Loss function used in the optimizer                                                                      # Name of dataset
    "input_embedding_size": 256,                                        # Size of input embedding layer
    "num_enc_layers": 3,                                         # Number of layers in the encoder
    "num_dec_layers": 3,                                         # Number of layers in the decoder
    "hidden_layer_size": 256,                                      # Size of hidden layer
    "dropout" : 0.30,                                            #Value of dropout used in  dropout
    'r_dropout':0.30,                                           # Value of dropout used in recurrent dropout
    "cell_type": 'GRU',                                         # Type of cell used in the encoder and decoder ('RNN' or 'GRU' or 'LSTM')
    "beam_width": 1                                          # Beam width used in beam decoder                                        
}
parser = argparse.ArgumentParser(description='Process the hyperparameters.')
parser.add_argument('-e','--epochs', type=type(config_['epochs']), nargs='?', default = config_['epochs']
                    ,help=f"Number of epochs(default {config_['epochs']})")
parser.add_argument('-o','--optimizer', type=type(config_['optimizer']), nargs='?', default = config_['optimizer']
                    ,help=f"Optimizer to be used for training the model( default {config_['optimizer']}) Allowed values : adam,sgd,nesterov,rmsprop,nadam,momentum")
parser.add_argument('-bs','--batch_size', type=type(config_['batch_size']), nargs='?', default = config_['batch_size']
                    ,help=f"Batch Size to be used(default {config_['batch_size']})")                    
parser.add_argument('-lr','--learning_rate', type=type(config_['learning_rate']), nargs='?', default = config_['learning_rate']
                    ,help=f"Learning rate of the model default( {config_['learning_rate']}")
parser.add_argument('-ies','--input_embedding_size', type=type(config_['input_embedding_size']), nargs='?', default = config_['input_embedding_size']
                    ,help=f"Size of input embedding layer default( {config_['input_embedding_size']})")
parser.add_argument('-nenc','--num_enc_layers', type=type(config_['num_enc_layers']), nargs='?', default = config_['num_enc_layers']
                    ,help=f"Number of layers in the encoder  default( {config_['num_enc_layers']})")
parser.add_argument('-ndec','--num_dec_layers', type=type(config_['num_dec_layers']), nargs='?', default = config_['num_dec_layers']
                    ,help=f"Number of layers in the decoder  default( {config_['num_dec_layers']})")
parser.add_argument('-hs','--hidden_layer_size', type=type(config_['hidden_layer_size']), nargs='?', default = config_['hidden_layer_size']
                    ,help=f"Size of hidden layer default( {config_['hidden_layer_size']})")
parser.add_argument('-d','--dropout', type=type(config_['dropout']), nargs='?', default = config_['dropout']
                    ,help=f"Value of dropout used in  dropout(default {config_['dropout']})")
parser.add_argument('-rd','--r_dropout', type=type(config_['r_dropout']), nargs='?', default = config_['r_dropout']
                    ,help=f"Value of dropout used in  recurrent dropout(default {config_['r_dropout']})")
parser.add_argument('-ct','--cell_type', type=type(config_['cell_type']), nargs='?', default = config_['cell_type']
                    ,help=f"tType of cell used in the encoder and decoder ('RNN' or 'GRU' or 'LSTM')(default {config_['cell_type']})")
parser.add_argument('-bw','--beam_width', type=type(config_['beam_width']), nargs='?', default = config_['beam_width']
                    ,help=f"Beam width used in beam decoder  default( {config_['beam_width']})")

args = parser.parse_args()
config_  = vars(args)
config_['loss_function'] = 'categorical_crossentropy'
print(config_)


def enc_dec_tokens(train_input_texts,train_target_texts,val_input_texts,val_target_texts):
    #Returns encoding of characters as integer in two dictionary for input and target characters
    #Returns number of tokens in input and output
    #Returns the maximum sequence length from input and target texts
    input_characters = parse_text(train_input_texts + val_input_texts)
    target_characters = parse_text(train_target_texts + val_target_texts)
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in train_input_texts + val_input_texts])
    max_decoder_seq_length = max([len(txt) for txt in train_target_texts + val_target_texts])

    print("Number of training samples:", len(train_input_texts))
    print("Number of validation samples:", len(val_input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)
    
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    
    return input_token_index,target_token_index,max_encoder_seq_length,max_decoder_seq_length,num_encoder_tokens,num_decoder_tokens

#Data Preprocessing
def data_processing(input_texts,enc_length,input_token_index,num_encoder_tokens, target_texts,dec_length,target_token_index,num_decoder_tokens):
    # Returns the input and target data in a form needed by the Keras embedding layer (i.e) 
    # decoder_input & encoder_input -- (None, timesteps) where each character is encoded by an integer
    # decoder_output -- (None, timesteps, vocabulary size) where the last dimension is the one-hot encoding

    # ' ' -- space (equivalent to no meaningful input / blank input)
    encoder_input_data = np.zeros(
        (len(input_texts), enc_length), dtype="float32"
    )
    decoder_input_data = np.zeros(
            (len(input_texts), dec_length), dtype="float32"
        )
    decoder_target_data = np.zeros(
            (len(input_texts), dec_length, num_decoder_tokens), dtype="float32"
        )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        
        for t, char in enumerate(input_text):
            encoder_input_data[i, t] = input_token_index[char]
        encoder_input_data[i, t + 1 :] = input_token_index[' ']
        
        for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t] = target_token_index[char]
            if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :] = target_token_index[' ']
        decoder_target_data[i, t:, target_token_index[' ']] = 1.0
    return encoder_input_data,decoder_input_data,decoder_target_data

input_token_index,target_token_index,max_encoder_seq_length,max_decoder_seq_length,num_encoder_tokens,num_decoder_tokens = enc_dec_tokens(train_input_texts,train_target_texts,val_input_texts,val_target_texts)
#Dictionary for reverse lookup of character for its integer encode 
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

#Preprocessed inputs this will be used in training validation and testing in future
train_encoder_input,train_decoder_input,train_decoder_target = data_processing(train_input_texts,max_encoder_seq_length,input_token_index,num_encoder_tokens, train_target_texts,max_decoder_seq_length,target_token_index,num_decoder_tokens)
val_encoder_input,val_decoder_input,val_decoder_target = data_processing(val_input_texts,max_encoder_seq_length,input_token_index,num_encoder_tokens, val_target_texts,max_decoder_seq_length,target_token_index,num_decoder_tokens)
test_encoder_input,test_decoder_input,test_decoder_target = data_processing(test_input_texts,max_encoder_seq_length,input_token_index,num_encoder_tokens, test_target_texts,max_decoder_seq_length,target_token_index,num_decoder_tokens)

from keras.layers import Layer
import keras.backend as K

class Attention(Layer):
    """
    This Attention layer class code is used from : https://github.com/thushv89/attention_keras/blob/master/src/layers/attention.py
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(Attention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            if verbose:
                print('Ws+Uh>', Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            # (batch_size, decoder_timesteps, decoder_hid_layer_size)
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            # (batch_size, decoder_timesteps, encoder_timesteps)
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

def make_model(num_encoder_tokens,num_decoder_tokens,input_embedding_size=64,num_enc_layers=1,num_dec_layers=1,hidden_layer_size=64,cell_type='LSTM',dropout=0,r_dropout=0,cell_activation='tanh'):
    '''
    Function to create a seq2seq model with attention.
    Arguments :
        num_encoder_tokens -- (int) number of characters in input vocabulary
        num_decoder_tokens -- (int) number of characters in output vocabulary
        input_embedding_size -- (int, default : 64) size of input embedding layer for encoder and decoder
        num_enc_layers -- (int, default : 1) number of layers of cell to stack in encoder
        num_dec_layers -- (int, default : 1) number of layers of cell to stack in decoder
        hidden_layer_size -- (int, default : 64) size of hidden layer of the encoder and decoder cells
        cell_type -- (string, default : 'LSTM') type of cell used in encoder and decoder (possible values : 'LSTM', 'GRU', 'RNN')
        dropout -- (float, default : 0.0) value of normal dropout (between 0 and 1)
        r_dropout -- (float, default : 0.0) value of recurrent dropout (between 0 and 1)
        cell_activation -- (string, default : 'tanh') type of activation used in the cell (as required by Keras)
    Returns :
        model -- (Keras model object) resulting attention model
    '''
    # Getting cell type
    cell = {
        'RNN':SimpleRNN,
        'LSTM':LSTM,
        'GRU':GRU
    }
     # Encoder input and embedding
    encoder_input = Input(shape=(None,),name='input_1')
    encoder_input_embedding = Embedding(num_encoder_tokens,input_embedding_size,name='embedding_1')(encoder_input)
    
    encoder_sequences, *encoder_state = cell[cell_type](hidden_layer_size,activation=cell_activation,return_sequences=True,return_state=True,dropout=dropout,recurrent_dropout=r_dropout,name="encoder_1")(encoder_input_embedding)
    # Encoder cell layers
    for i in range(1,num_enc_layers):
        encoder_sequences, *encoder_state = cell[cell_type](hidden_layer_size,activation=cell_activation,return_sequences=True,return_state=True,dropout=dropout,recurrent_dropout=r_dropout,name=f"encoder_{i+1}")(encoder_sequences)
    
    # Decoder input and embedding
    decoder_input = Input(shape=(None,),name='input_2')
    decoder_input_embedding = Embedding(num_decoder_tokens,input_embedding_size,name='embedding_2')(decoder_input)
    
    decoder_sequences = decoder_input_embedding
    # Decoder cell layers
    for i in range(num_dec_layers-1):
        decoder_sequences, *decoder_state = cell[cell_type](hidden_layer_size,activation=cell_activation,return_sequences=True,return_state=True,dropout=dropout,recurrent_dropout=r_dropout,name=f"decoder_{i+1}")(decoder_sequences ,initial_state=encoder_state)
    # Decoder last layer
    decoder_sequences, *decoder_state = cell[cell_type](hidden_layer_size,activation=cell_activation,return_sequences=True,return_state=True,dropout=dropout,recurrent_dropout=r_dropout,name="decoder_1")(decoder_input_embedding ,initial_state=encoder_state)
    
    # Attention layer
    attention_out,attention_scores = Attention(name="attention_1")([encoder_sequences,decoder_sequences])
    
    # Concat attention output and decoder output
    dense_concat_input = keras.layers.Concatenate(axis=-1,name='concat_layer_1')([decoder_sequences,attention_out])
    # Time distributed Softmax FC layer
    decoder_dense = Dense(num_decoder_tokens,activation="softmax",name="dense_1")(dense_concat_input)
    
    # Define the model that will turn encoder_input_data and decoder_input_data into decoder_target_data
    model = keras.Model([encoder_input,decoder_input],decoder_dense)
    model.summary()
    return model

def make_inference_model(model):
    '''
    Function to return models needed for inference from the original model (with attention).
    Arguments :
        model -- (Keras model object) attention model used for training
    Returns :
        encoder_model -- (Keras model object) 
        deocder_model -- (Keras model object)
        num_enc_layers -- (int) number of layers in the encoder
        num_dec_layers -- (int) number of layers in the decoder
    '''
    # Calculating number of layers in encoder and decoder
    num_enc_layers, num_dec_layers = 0, 0
    for layer in model.layers:
        num_enc_layers += layer.name.startswith('encoder')
        num_dec_layers += layer.name.startswith('decoder')

    # Encoder input
    encoder_input = model.input[0]      # Input_1
    # Encoder cell final layer
    encoder_cell = model.get_layer("encoder_"+str(num_enc_layers))
    encoder_type = encoder_cell.__class__.__name__
    encoder_sequences, *encoder_state = encoder_cell.output
    # Encoder model
    encoder_model = keras.Model(encoder_input, encoder_state)

    # Decoder input
    decoder_input = model.input[1]      # Input_2
    decoder_input_embedding = model.get_layer("embedding_2")(decoder_input)
    decoder_sequences = decoder_input_embedding
    # Inputs to decoder layers' initial states
    decoder_states, decoder_state_inputs = [], []
    for i in range(1, num_dec_layers+1):
        if encoder_type == 'LSTM':
            decoder_state_input = [Input(shape=(encoder_state[0].shape[1],), name="input_"+str(2*i+1)), 
                                   Input(shape=(encoder_state[1].shape[1],), name="input_"+str(2*i+2))]
        else:
            decoder_state_input = [Input(shape=(encoder_state[0].shape[1],), name="input_"+str(i+2))]

        decoder_cell = model.get_layer("decoder_"+str(i))
        decoder_sequences, *decoder_state = decoder_cell(decoder_sequences, initial_state=decoder_state_input)
        decoder_states += decoder_state
        decoder_state_inputs += decoder_state_input
    # Attention layer
    attention_out,attention_scores = model.get_layer("attention_1")([encoder_sequences,decoder_sequences])
    
    dense_concat_input = keras.layers.Concatenate(axis=-1,name='concat_layer_1')([decoder_sequences,attention_out])
    # Softmax FC layer
    decoder_dense = model.get_layer("dense_1")
    decoder_dense_output = decoder_dense(dense_concat_input)

    # Decoder model
    decoder_model = keras.Model(
        [encoder_input,decoder_input] + decoder_state_inputs, [attention_scores,decoder_dense_output] + decoder_states
    )

    return encoder_model, decoder_model, num_enc_layers, num_dec_layers


def num_to_word(num_encoded, token_index, reverse_char_index = None):
    # Function to return the predictions after cutting the '\n' and ' ' s at the end.
    # If reverse_char_index == None, the predictions are in the form of decoded string, otherwise as list of integers
    num_samples = len(num_encoded) if type(num_encoded) is list else num_encoded.shape[0]
    predicted_words = ['' for t in range(num_samples)]
    for i, encode in enumerate(num_encoded):
        for l in encode:
            # Stop word : '\n'
            if l == token_index['\n']:
                break
            predicted_words[i] += reverse_char_index[l] if reverse_char_index is not None else str(l)
    
    return predicted_words

def beam_decoder_util(model,input_sequences,max_decoder_seq_length,B=1,target_sequences=None,start_char=0,batch_size=64):
    '''
    Function to do inference on the model using beam decoder.
    Arguments :
        model -- (Keras model object) training model
        input_sequences -- (numpy ndarray of size : (None, timesteps)) input to encoder
        max_decoder_seq_length -- (int) Number of timesteps to infer in decoder
        B -- (int, default : 1) beam width of beam decoder
        target_sequences -- (numpy ndarray of size : (None, timesteps, num_decoder_tokens), deault : None) expected target.
                       If None, cross entropy errors won't be calculated.
        start_char -- (int, default : 0) Encoding integer for ' '(start char)
        batch_size -- (int, default : 64) batch_size sent to Keras predict
    Returns :
        final_outputs -- (numpy ndarray of size : (None, B, timesteps)) top B output sequences
        final_errors -- (numpy ndarray of size : (None, B)) cross entropy errors for top B output (All zeros if target_seqs == None)
        states_values -- (numpy ndarray of size : (, None, timesteps, hidden_layer_size))  hidden states of decoder
        final_attn_scores -- (numpy ndarray of size : (None, B, decoder_timesteps(max_decoder_seq_length), encoder_timesteps(max_encoder_seq_length))) attention to all encoder timesteps for a decoder timestep 
    '''
    # Generating output from encoder
    encoder_model,decoder_model,num_enc_layers,num_dec_layers=make_inference_model(model)
    encoder_output = encoder_model.predict(input_sequences,batch_size=batch_size)
    encoder_output = encoder_output if type(encoder_output) is list else [encoder_output]
    # Number of input samples in the data passed
    num_samples = input_sequences.shape[0]
    # Top B output sequences for each input 
    outputs_fn = np.zeros((num_samples,B,max_decoder_seq_length),dtype=np.int32)
    # Errors for top B output sequences for each input
    errors_fn = np.zeros((num_samples,B))
    
    # decoder input sequence for 1 timestep (for all samples). Initially one choice only there
    decoder_b_inputs = np.zeros((num_samples,1,1))
    # Populate the input sequence with the start character at the 1st timestep
    decoder_b_inputs[:, :, 0] = start_char
    
    # (log(probability) sequence, decoder output sequence) pairs for all choices and all samples. Probability starts with log(1) = 0
    decoder_b_out = [[(0, [])] for t in range(num_samples)]
    # Categorical cross entropy error in the sequence for all choice and all samples
    errors = [[0] for t in range(num_samples)]
    # Output states from decoder for all choices, and all samples
    states = [encoder_output*num_dec_layers]
    # Attention weights output
    attn_b_scores = [[None] for t in range(num_samples)]
    
    # Sampling loop
    for idx in range(max_decoder_seq_length):
        # Storing respective data for all possibilities
        all_b_beams = [[] for t in range(num_samples)]
        all_decoder_states = [[] for t in range(num_samples)]
        all_errors = [[] for t in range(num_samples)]
        all_attn_scores = [[] for t in range(num_samples)]
        for b in range(len(decoder_b_out[0])):
            attn_scores,decoder_output, *decoder_states = decoder_model.predict([input_sequences,decoder_b_inputs[:,b]] + states[b],batch_size=batch_size)
            # Top B scores
            top_b = np.argsort(decoder_output[:,-1,:],axis=-1)[:,-B:]
            for n in range(num_samples):
                all_b_beams[n]+= [(decoder_b_out[n][b][0] + np.log(decoder_output[n, -1, top_b[n][i]]),decoder_b_out[n][b][1] + [top_b[n][i]]) for i in range(B)]
                all_attn_scores[n] += [attn_scores[n]]*B if attn_b_scores[n][b] is None else [np.concatenate((attn_b_scores[n][b],attn_scores[n]),axis=0)]*B
                if target_sequences is not None:
                    all_errors[n] += [errors[n][b] - np.log(decoder_output[n,-1,target_sequences[n,idx]])]*B
                all_decoder_states[n] += [[decoder_state[n:n+1] for decoder_state in decoder_states]] * B
        # Sort and choose top B with max probabilities
        sorted_index = list(range(len(all_b_beams[0])))
        sorted_index = [sorted(sorted_index,key = lambda ix: all_b_beams[n][ix][0])[-B:][::-1] for n in range(num_samples)]
        # Choose the top B decoder output sequences till now
        decoder_b_out = [[all_b_beams[n][index] for index in sorted_index[n]] for n in range(num_samples)]
        
        # Update the input sequence for next 1 timestep
        decoder_b_inputs = np.array([[all_b_beams[n][index][1][-1] for index in sorted_index[n]] for n in range(num_samples)])
        # Update states
        states = [all_decoder_states[0][index] for index in sorted_index[0]]
        for n in range(1,num_samples):
            states = [[np.concatenate((states[i][j],all_decoder_states[n][index][j])) for j in range(len(all_decoder_states[n][index]))] for i,index in  enumerate(sorted_index[n])]
        # Update attention scores
        attn_b_scores = [[all_attn_scores[n][index] for index in sorted_index[n]] for n in range(num_samples)]    
        if target_sequences is not None:
            errors = [[all_errors[n][index] for index in sorted_index[n]] for n in range(num_samples)]
            
    outputs_fn = np.array([[decoder_b_out[n][i][1] for i in range(B)] for n in range(num_samples)])
    # Update errors
    if target_sequences is not None:
        errors_fn = np.array(errors)/max_decoder_seq_length
    return outputs_fn,errors_fn,np.array(states),np.array(attn_b_scores)

def calc_metrics(b_outputs, target_sequences,token_index,reverse_char_index,b_errors=None,exact_word=True,display=False):
    # Calculates the accuracy (and mean error if info provided) for the best of B possible output sequences
    # target_sequencess -- Expected output (encoded sequence)
    # b_outputs -- b choices of output sequences for each sample
    matches = np.mean(b_outputs == target_sequences.reshape((target_sequences.shape[0],1,target_sequences.shape[1])),axis=-1)
    best_b = np.argmax(matches,axis=-1)
    best_index = (tuple(range(best_b.shape[0])),tuple(best_b))
    accuracy = np.mean(matches[best_index])
    b_predictions = list()
    loss = None
    if b_errors is not None:
        loss = np.mean(b_errors[best_index])
    if exact_word:
        equal = [0] * b_outputs.shape[0]
        true_out = num_to_word(target_sequences,token_index,reverse_char_index)
        for b in range(b_outputs.shape[1]):
            pred_out = num_to_word(b_outputs[:,b], token_index,reverse_char_index)
            equal = [equal[i] or (pred_out[i] == true_out[i]) for i in range(b_outputs.shape[0])]
            if display==True:
                b_predictions.append(pred_out)
        exact_accuracy = np.mean(equal)
        if display==True:
            return accuracy,exact_accuracy,loss,true_out,b_predictions
        return accuracy,exact_accuracy,loss
    return accuracy,loss
def beam_decoder(model,input_sequences,target_sequences_onehot,max_decoder_seq_length,token_index,reverse_char_index,B=1,model_batch_size=64,infer_batch_size=512,exact_word=True,return_outputs=False,return_states=False,return_attention=False,display=False):
    '''
    Function to calculate/capture character-wise accuracy, exact-word-match accuracy, and loss for the seq2seq model using a beam decoder.
    Arguments :
        model -- (Keras model object) model used for training
        input_sequences -- (numpy ndarray of size : (None, timesteps)) input to encoder (where characters are encoded as integers)
        target_sequences -- (numpy ndarray of size : (None, timesteps, num_decoder_tokens)) expected target in onehot format
        max_decoder_seq_length -- (int) Number of timesteps to infer in decoder
        token_index -- (dict) target character encoding
        reverse_char_index -- (dict) target character decoding
        B -- (int, default : 1) beam width to be used in beam decoder
        model_batch_size -- (int, default : 64) batch size to be used while evaluating model using Keras
        infer_batch_size -- (int, default : 512) number of samples to be sent to beam_decoder_infer() at a time (to avoid RAM memory overshoot problems).
                            We have set the default model_batch_size and infer_batch_size such that it takes the least time to run and runs without problems.
        exact_word -- (bool, default : True) whether or not exact_accuracy has (If True, will be returned as the next argument after accuracy)
        return_outputs -- (bool, default : True) whether or not the outputs predicted need to be returned
        return_states -- (bool, default : True) whether or not the decoder hidden states need to be returned (for further training, another sequential model addition, etc)
        return_attn_scores -- (bool, default : True) whether or not the attention scores need to be returned
    Returns :
        accuracy -- (float) the character-wise match accuracy (as calculated by Keras fit)
        (If exact_word is True) exact_accuracy -- (float) the exact word match accuracy
        loss -- (float) the cross-entropy loss for the top B predictions
        (If return_outputs is True) b_outputs -- (numpy ndarray of size : (None, B, timesteps)) top B output sequences
        (If return_states is True) b_states -- (numpy ndarray of size : (B, None, timesteps, hidden_layer_size))  hidden states of decoder
        (If return_attn_scores is True) b_attn_scores -- (numpy ndarray of size : (None, B, decoder_timesteps, encoder_timesteps)) attention scores
    '''
    target_sequences = np.argmax(target_sequences_onehot,axis=-1)
    b_outputs,b_errors,b_states,b_attention=None,None,None,None
    for i in range(0,input_sequences.shape[0],infer_batch_size):
        tmp_b_outputs,tmp_b_errors,tmp_b_states,tmp_b_attention = beam_decoder_util(model,input_sequences[i:i+infer_batch_size],max_decoder_seq_length,B,target_sequences[i:i+infer_batch_size],token_index['\t'],model_batch_size)
        
        if b_errors is None:
            b_outputs,b_errors,b_states,b_attention = tmp_b_outputs,tmp_b_errors,tmp_b_states,tmp_b_attention
        else:
            b_outputs = np.concatenate((b_outputs,tmp_b_outputs))
            b_errors = np.concatenate((b_errors,tmp_b_errors))
            b_states = np.concatenate((b_states,tmp_b_states),axis=2)
            b_attention = np.concatenate((b_attention,tmp_b_attention))
    return_elements = []
    if return_outputs:
        return_elements += [b_outputs]
    if return_states:
        return_elements += [b_states]
    if return_attention:
        return_elements += [b_attention]
    if len(return_elements) > 0:
        return calc_metrics(b_outputs,target_sequences,token_index,reverse_char_index,b_errors,exact_word,display) + tuple(return_elements)
    return calc_metrics(b_outputs,target_sequences,target_token_index,reverse_char_index,b_errors,exact_word,display)


def model_train_util(config):
    #utility function to build and compile model based on the configuration passed as input and return the model
    model = make_model(num_encoder_tokens,num_decoder_tokens,config['input_embedding_size'],config['num_enc_layers'],config['num_dec_layers'],config['hidden_layer_size'],config['cell_type'],config['dropout'],config['r_dropout'])
    optimizer = config['optimizer']
    if config['loss_function'] == 'categorical_crossentropy':
        loss_fn = CategoricalCrossentropy
    else:
        loss_fn = SparseCategoricalCrossentropy
    if optimizer == 'adam':
        model.compile(optimizer = Adam(learning_rate=config['learning_rate']), loss = loss_fn(), metrics = ['accuracy'])
    elif optimizer == 'momentum':
        model.compile(optimizer = SGD(learning_rate=config['learning_rate'], momentum = 0.9), loss = loss_fn(), metrics = ['accuracy'])
    elif optimizer == 'rmsprop':
        model.compile(optimizer = RMSprop(learning_rate=config['learning_rate']), loss = loss_fn(), metrics = ['accuracy'])
    elif optimizer == 'nesterov':
        model.compile(optimizer = SGD(learning_rate=config['learning_rate'], momentum = 0.9, nesterov = True), loss = loss_fn(), metrics = ['accuracy'])
    elif optimizer == 'nadam':
        model.compile(optimizer = Nadam(learning_rate=config['learning_rate']), loss = loss_fn(), metrics = ['accuracy'])
    else:
        model.compile(optimizer = SGD(learning_rate=config['learning_rate']), loss = loss_fn(), metrics = ['accuracy'])
    
    return model

class customCallback(keras.callbacks.Callback):
     # Custom class to provide callback after each epoch of training to calculate custom metrics for validation set with beam decoder
    def __init__(self, val_enc_input, val_dec_target, beam_width=1, batch_size=64):
        self.beam_width = beam_width
        self.validation_input = val_enc_input
        self.validation_target = val_dec_target
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs):
        val_accuracy, val_exact_accuracy, val_loss = beam_decoder(self.model, self.validation_input, self.validation_target, max_decoder_seq_length, 
                                                                  target_token_index, reverse_target_char_index, self.beam_width, self.batch_size)

        # Log them to reflect in WANDB callback and EarlyStopping
        logs["val_accuracy"] = val_accuracy
        logs["val_exact_accuracy"] = val_exact_accuracy
        logs["val_loss"] = val_loss             # Validation loss calculates categorical cross entropy loss

        print("— val_loss: {:.3f} — val_accuracy: {:.3f} — val_exact_accuracy: {:.5f}".format(val_loss, val_accuracy, val_exact_accuracy))



def model_train(config,iswandb=False):
    '''
    Arguments:
        config -- (dict) Hyperparameter config with which model is trained
        iswandb -- (bool,default:False) If True training is done by logging metrics to wandb otherwise training is done as it is
    Returns:
        wid: Wandb run id if any 
        model - Keras model
        history - metrics of training
        config - Same as passed in argument
    '''
    
    wid = None
    if iswandb:
        wid = wandb.util.generate_id()
        run = wandb.init(id = wid, project="cs6910_assignment_3_attention", entity="dlstack", reinit=True, config=config)
        wandb.run.name = f"ies_{config['input_embedding_size']}_nenc_{config['num_enc_layers']}_ndec_{config['num_dec_layers']}_cell_{config['cell_type']}_drop_{config['dropout']}_rdrop{config['r_dropout']}"
        wandb.run.name += f"_hs_{config['hidden_layer_size']}_B_{config['beam_width']}_attn"
        wandb.run.save()
        print(wandb.run.name)

    model = model_train_util(config)
    if iswandb:
        call_list = [customCallback(val_encoder_input,val_decoder_target,beam_width=config['beam_width'],batch_size=config['batch_size']),WandbCallback(monitor='val_accuracy'),EarlyStopping(monitor='val_accuracy',patience=4)]
    else:
        call_list = [customCallback(val_encoder_input,val_decoder_target,beam_width=config['beam_width'],batch_size=config['batch_size']),EarlyStopping(monitor='val_accuracy',patience=4)]
    history = model.fit(
        [train_encoder_input,train_decoder_input],
        train_decoder_target,
        batch_size=config['batch_size'],
        verbose = 1,
        epochs=config['epochs'],
        callbacks = call_list
    )    
    if iswandb:
        run.finish()

    return model, history,config, wid


def wandb_sweep():
    # Wrapper function to call the model_train() function for sweeping with different hyperparameters

    # Initialize a new wandb run
    run = wandb.init(config=config_, reinit=True)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    wandb.run.name = f'ies_{config.input_embedding_size}_nenc_{config.num_enc_layers}_ndec_{config.num_dec_layers}_cell_{config.cell_type}_drop_{config.dropout}_rdrop_{config.r_dropout}'
    wandb.run.name += f'_hs_{config.hidden_layer_size}_B_{config.beam_width}_attn'
    wandb.run.save()
    print(wandb.run.name)

    model, *_ = model_train(config, iswandb=True)
    run.finish()

model,history,config,_ = model_train(config_)
model.save("s2s")

###UNCOMMENT TO RUN SWEEPS WITH WANDB####
# Hyperparameter choices to sweep 
# sweep_config_1 = {
#     'name': 'RNNs2s_attn',
#     'method': 'bayes',                   # Possible search : grid, random, bayes
#     'metric': {
#       'name': 'val_accuracy',
#       'goal': 'maximize'   
#     },
#     'parameters': {
#         'epochs':{
#             'values':[10,15,20]
#         },
#         'learning_rate':{
#             'values':[0.001,0.0001,0.005]
#         },
#         'optimizer':{
#             'value':'adam'
#         },
#         'loss_function':{
#           'value':'categorical_crossentropy'  
#         },
#         'input_embedding_size': {
#             'values': [64,128,256]
#         },
#         'num_enc_layers': {
#             'values': [2,3]
#         },
#         'num_dec_layers': {
#             'values': [3,5]
#         },
#         'hidden_layer_size': {
#             'values': [256,512,768]
#         },
#         'cell_type': {
#             'values': ['GRU']
#         },
#         'dropout' :{
#             'values': [0.20,0.30]
#         },
#         'r_dropout': {
#             'values': [0.20,0.30]
#         },
#         'beam_width': {
#             'values': [1,3,5]
#         },
#         'batch_size':{
#             'values':[128,256]
#         }
#     }
# }
# sweep_id = wandb.sweep(sweep_config_1,entity='dlstack',project='cs6910_assignment_3_attention')
# wandb.agent(sweep_id,lambda:wandb_sweep(),count=10)


# # In[ ]:


# # Hyperparameter choices to sweep 
# sweep_config_1 = {
#     'name': 'RNNs2s_attn',
#     'method': 'bayes',                   # Possible search : grid, random, bayes
#     'metric': {
#       'name': 'val_accuracy',
#       'goal': 'maximize'   
#     },
#     'parameters': {
#         'epochs':{
#             'values':[10,15,20]
#         },
#         'learning_rate':{
#             'values':[0.001,0.0001,0.005]
#         },
#         'optimizer':{
#             'value':'adam'
#         },
#         'loss_function':{
#           'value':'categorical_crossentropy'  
#         },
#         'input_embedding_size': {
#             'values': [64,128,256]
#         },
#         'num_enc_layers': {
#             'values': [2,3]
#         },
#         'num_dec_layers': {
#             'values': [3,5]
#         },
#         'hidden_layer_size': {
#             'values': [256,512,768]
#         },
#         'cell_type': {
#             'values': ['GRU']
#         },
#         'dropout' :{
#             'values': [0.20,0.30]
#         },
#         'r_dropout': {
#             'values': [0.20,0.30]
#         },
#         'beam_width': {
#             'values': [1,3,5]
#         },
#         'batch_size':{
#             'values':[128,256]
#         }
#     }
# }
# sweep_id = wandb.sweep(sweep_config_1,entity='dlstack',project='cs6910_assignment_3_attention')
# wandb.agent(sweep_id,lambda:wandb_sweep(),count=10)


# # In[ ]:


# # Hyperparameter choices to sweep 
# sweep_config_1 = {
#     'name': 'RNNs2s_attn',
#     'method': 'bayes',                   # Possible search : grid, random, bayes
#     'metric': {
#       'name': 'val_accuracy',
#       'goal': 'maximize'   
#     },
#     'parameters': {
#         'epochs':{
#             'values':[10,15,20]
#         },
#         'learning_rate':{
#             'values':[1e-3,1e-4]
#         },
#         'optimizer':{
#             'values':['rmsprop','adam','nadam','nesterov','sgd']
#         },
#         'loss_function':{
#           'value':'categorical_crossentropy'  
#         },
#         'input_embedding_size': {
#             'values': [32, 64,256]
#         },
#         'num_enc_layers': {
#             'values': [ 2, 3,4]
#         },
#         'num_dec_layers': {
#             'values': [ 2, 3,4]
#         },
#         'hidden_layer_size': {
#             'values': [64, 128, 256,512]
#         },
#         'cell_type': {
#             'values': ['RNN', 'LSTM', 'GRU']
#         },
#         'dropout' :{
#             'values': [0, 0.25, 0.3,0.4]
#         },
#         'r_dropout':{
#           'values':[0.0,0.1]  
#         },
#         'batch_size':{
#           'values':[64,128,256]  
#         },
#         'beam_width': {
#             'values': [1, 5]
#         }
#     }
# }
# sweep_id = wandb.sweep(sweep_config_1,entity='dlstack',project='cs6910_assignment_3_attention')
# wandb.agent(sweep_id,lambda:wandb_sweep(),count=10)

