
# **Installing dependencies**
# !pip install wget
# !pip install --upgrade wandb

# **Importing necessary libraries,packages,etc.**

from __future__ import print_function
import wget
import os
import tarfile
import csv
import matplotlib
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from ipywidgets import interact, Layout, IntSlider
from IPython.display import HTML as html_print
from IPython.display import display
import random

import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model

import wandb
from wandb.keras import WandbCallback
wandb.login(key='b44266d937596fcef83bedbe7330d6cee108a277')

from keras.layers import SimpleRNN,GRU,LSTM,Embedding,Input,Dense
from keras.layers import Layer
import keras.backend as K

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

        #  ' ' -- space (equivalent to no meaningful input / blank input)
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

#Inference Model for attention
def make_inference_model_attn(model):
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
    # Attention laye
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
#Inference model without attention
def make_inference_model(model):
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
    encoder_seq, *encoder_state = encoder_cell.output
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

    # Softmax FC layer
    decoder_dense = model.get_layer("dense_1")
    decoder_dense_output = decoder_dense(decoder_sequences)

    # Decoder model
    decoder_model = keras.Model(
        [decoder_input] + decoder_state_inputs, [decoder_dense_output] + decoder_states
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

def beam_decoder_util(model,input_sequences,max_decoder_seq_length,B=1,target_sequences=None,start_char=0,batch_size=64,attention=False):
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
    if attention:
        encoder_model,decoder_model,num_enc_layers,num_dec_layers=make_inference_model_attn(model)
    else:
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
    if attention:
        # Attention weights output
        attn_b_scores = [[None] for t in range(num_samples)]
    # Sampling loop
    for idx in range(max_decoder_seq_length):
        # Storing respective data for all possibilities
        all_b_beams = [[] for t in range(num_samples)]
        all_decoder_states = [[] for t in range(num_samples)]
        all_errors = [[] for t in range(num_samples)]
        if attention:
            all_attn_scores = [[] for t in range(num_samples)]
        for b in range(len(decoder_b_out[0])):
            if attention:
                attn_scores,decoder_output, *decoder_states = decoder_model.predict([input_sequences,decoder_b_inputs[:,b]] + states[b],batch_size=batch_size)
            else:
                decoder_output, *decoder_states = decoder_model.predict([decoder_b_inputs[:,b]] + states[b],batch_size=batch_size)
            # Top B scores
            top_b = np.argsort(decoder_output[:,-1,:],axis=-1)[:,-B:]
            for n in range(num_samples):
                all_b_beams[n]+= [(decoder_b_out[n][b][0] + np.log(decoder_output[n, -1, top_b[n][i]]),decoder_b_out[n][b][1] + [top_b[n][i]]) for i in range(B)]
                if attention:
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
            
        if attention:
            # Update attention scores
            attn_b_scores = [[all_attn_scores[n][index] for index in sorted_index[n]] for n in range(num_samples)]    
        # Update errors 
        if target_sequences is not None:
            errors = [[all_errors[n][index] for index in sorted_index[n]] for n in range(num_samples)]
    
           
    outputs_fn = np.array([[decoder_b_out[n][i][1] for i in range(B)] for n in range(num_samples)])
    if target_sequences is not None:
        errors_fn = np.array(errors)/max_decoder_seq_length
        if attention:
            return outputs_fn,errors_fn,np.array(states),np.array(attn_b_scores)
        return outputs_fn,errors_fn,np.array(states)

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

def beam_decoder(model,input_sequences,target_sequences_onehot,max_decoder_seq_length,token_index,reverse_char_index,B=1,model_batch_size=64,infer_batch_size=512,exact_word=True,attention=False,return_outputs=False,return_states=False,return_attention=False,display=False):
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
        attention -- (bool, defualt : False) whether the model has attention or not
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
    if attention:
        b_outputs,b_errors,b_states,b_attention=None,None,None,None
    else:
        b_outputs,b_errors,b_states=None,None,None
    for i in range(0,input_sequences.shape[0],infer_batch_size):
        if attention:
            tmp_b_outputs,tmp_b_errors,tmp_b_states,tmp_b_attention = beam_decoder_util(model,input_sequences[i:i+infer_batch_size],max_decoder_seq_length,B,target_sequences[i:i+infer_batch_size],token_index['\t'],model_batch_size,attention=True)
        else:
            tmp_b_outputs,tmp_b_errors,tmp_b_states = beam_decoder_util(model,input_sequences[i:i+infer_batch_size],max_decoder_seq_length,B,target_sequences[i:i+infer_batch_size],token_index['\t'],model_batch_size,attention=False)
        if b_errors is None:
            if attention:
                b_outputs,b_errors,b_states,b_attention = tmp_b_outputs,tmp_b_errors,tmp_b_states,tmp_b_attention
            else:
                b_outputs,b_errors,b_states = tmp_b_outputs,tmp_b_errors,tmp_b_states
        else:
            if attention:
                b_outputs = np.concatenate((b_outputs,tmp_b_outputs))
                b_errors = np.concatenate((b_errors,tmp_b_errors))
                b_states = np.concatenate((b_states,tmp_b_states),axis=2)
                b_attention = np.concatenate((b_attention,tmp_b_attention))
            else:
                b_outputs = np.concatenate((b_outputs,tmp_b_outputs))
                b_errors = np.concatenate((b_errors,tmp_b_errors))
                b_states = np.concatenate((b_states,tmp_b_states),axis=2) 
    return_elements = []
    if return_outputs:
        return_elements += [b_outputs]
    if return_states:
        return_elements += [b_states]
    if return_attention and attention:
        return_elements += [b_attention]
    if len(return_elements) > 0:
        return calc_metrics(b_outputs,target_sequences,token_index,reverse_char_index,b_errors,exact_word,display) + tuple(return_elements)
    return calc_metrics(b_outputs,target_sequences,target_token_index,reverse_char_index,b_errors,exact_word,display)

def levenshtein(s1, s2):
    # Function to calculate levenshtein distance between two sequences usign Dynamic Programming
    m, n = len(s1)+1, len(s2)+1
    # Initialisation
    dp = np.zeros((m, n))
    # Base case
    dp[0,1:] = np.arange(1,n)
    dp[1:,0] = np.arange(1,m)

    # Recursion
    for i in range(1,m):
        for j in range(1,n):
            if s1[i-1] == s2[j-1]:
                dp[i,j] = min(dp[i-1,j-1], dp[i-1,j]+1, dp[i,j-1]+1)
            else:
                dp[i,j] = min(dp[i,j-1], dp[i-1,j], dp[i-1,j-1]) + 1
    
    return dp[m-1,n-1]

def test_model(run_id,test_encoder_input,test_decoder_target,max_decoder_seq_length,target_token_index,reverse_target_char_index,attention=False,save_pred=False,test_input_texts=None):
    '''
    Function to evaluate the model metrics on test data and optionally save the predictions.
    Arguments :
        run_id -- (string) WANDB run ID for the trained model
        test_encoder_input -- (numpy ndarray of size : (None, timesteps)) input to encoder (where characters are encoded as integers)
        test_decoder_target -- (numpy ndarray of size : (None, timesteps, decoder_vocab_size)) expected target in onehot format
        max_decoder_seq_length -- (int) number of timesteps in the decoder
        target_token_index -- (dict) target character encoding
        reverse_target_char_index -- (dict) target character decoding
        attention -- (bool, default : False) whether or not the model uses attention
        save_pred -- (bool, default : False) whether or not to save the predictions in a csv file
        test_input_texts -- (list of string : (no_samples, input word), default : None) input as words (needed while saving predictions to file alone)
    Returns :
        acc -- (float) character-wise match accuracy
        exact_B_acc -- (float) exact word match accuracy using the beam width for the model
        exact_acc -- (float) exact word match accuracy using the first prediction (which is equivalent to beam width = 1)
        loss -- (float) loss value
        true_out -- (list of string : (no_samples, word)) true output  
        pred_out -- (2D list of string : (no_samples, B, word)) predicted output
        pred_scores -- (2D list : (no_samples, B)) levenshtein distance of prediction to true output
        (If attention is True) attn_scores -- (numpy ndarray of size : (None, B, decoder_timesteps, encoder_timesteps)) attention scores
        model -- (Keras model object) the model obtained from the run
    '''
    api = wandb.Api()
    r = api.run('dlstack/cs6910_assignment_3_attention/'+run_id) if attention else api.run('dlstack/cs6910_assignment_3/'+run_id)
    config = r.config['_items'] if '_items' in r.config.keys() else r.config
    model_file = r.file('model-best.h5').download(replace=True)
    if attention:
        model = keras.models.load_model(model_file.name,custom_objects={'Attention':Attention})
    else:
        model = keras.models.load_model(model_file.name)
    
    num_samples,batch_size,B = test_encoder_input.shape[0],config['batch_size'],config['beam_width']
    if attention:
        acc, exact_B_acc, loss, outputs,attention_scores = beam_decoder(model, test_encoder_input, test_decoder_target, max_decoder_seq_length, 
                                                                target_token_index, reverse_target_char_index,B, batch_size,attention=True,
                                                                return_outputs=True,return_attention=True)
    else:
        acc, exact_B_acc, loss, outputs = beam_decoder(model, test_encoder_input, test_decoder_target, max_decoder_seq_length, 
                                                                target_token_index, reverse_target_char_index,B, batch_size,attention=False,
                                                                return_outputs=True,return_attention=False)    
    print(f'Test accuracy (using exact word match with beam width = {B}) : {exact_B_acc*100:.2f}%')
    
    test_target = np.argmax(test_decoder_target, axis=-1)
    true_out = num_to_word(test_target, target_token_index, reverse_target_char_index)
    pred_out = [[] for t in range(num_samples)]
    pred_scores = [[] for t in range(num_samples)]
    for b in range(B):
        pred = num_to_word(outputs[:,b], target_token_index, reverse_target_char_index)
        pred_out = [pred_out[n] + [pred[n]] for n in range(num_samples)]
        pred_scores = [pred_scores[n] + [levenshtein(pred[n], true_out[n])] for n in range(num_samples)]

    equal = [pred_out[n][0] == true_out[n] for n in range(num_samples)]
    exact_acc = np.mean(equal)

    print(f'Test accuracy (using exact word match of the first prediction) : {exact_acc*100:.2f}%')
    print('\n')
    save_pred = True
    if save_pred:
        # We write the input and top K outputs in decreasing order of probabilities to the file
        pred_file_name = 'predictions_attention.csv' if attention else 'predictions.csv'
        with open(pred_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Input"] + ["Prediction_"+str(b+1) for b in range(B)])
            for n in range(num_samples):
                writer.writerow([test_input_texts[n]] + [pred_out[n][b] for b in range(B)])
    if attention:
        return acc, exact_B_acc, exact_acc, loss, true_out, pred_out, pred_scores, attention_scores, model
    return acc, exact_B_acc, exact_acc, loss, true_out, pred_out, pred_scores, model

def get_clr(value, cmap=None):
    # Function to get appropriate color for a value between 0 and 1 from the default blue to red hard-coded colors or a matplotlib cmap 
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8',
        '#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
        '#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
        '#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
    if cmap is not None:
        rgba = matplotlib.cm.get_cmap(cmap)(value,alpha=None,bytes=True)
        return 'rgb'+str(rgba[:-1])
    value = min(int((value * 100) / 5), 19)
    return colors[value]

def print_samples(inputs, true_output, pred_output, pred_scores, attention=False,wandb_log=False,random_seq=None,cmap=None):
    '''
    Function to print sample outputs in a neat format
    Arguments :
        input -- input words
        true_output -- true output as words
        pred_output -- B predicted output words
        pred_scores -- levenshtein distance for the predictions to the true output
        attention -- (bool, default : False) whether or not the model uses attention
        wandb_log -- (bool, default: False) whether or not to log the predictions to wandb as html file
        random_seq -- list of indices from the dataset passed for which the sample outputs are to be printed (If None, random 10 samples will be chosen)
    Returns :
        random_seq -- the list of indices for which sample outputs are printed
    '''
    num_samples = len(true_output)
    if random_seq is None:
        random_seq = random.sample(range(num_samples),10)
    if attention:
        headline = '-'*20 + f' Top {len(pred_scores[0])} predictions with attention in decreasing order of probabilities for 10 random samples ' + '-'*20
    else:
        headline = '-'*20 + f' Top {len(pred_scores[0])} predictions in decreasing order of probabilities for 10 random samples ' + '-'*20
    print(headline)
    print('')
    html_body=''
    for i in random_seq:
        K = len(pred_scores[i])
        html_str = '''
        <table style="border:2px solid black; border-collapse:collapse">
        <caption> <strong>INPUT :</strong> {} &emsp; | &emsp; <strong> TRUE OUTPUT : </strong> {} </caption>
        <tr>
        <th scope="row" style="border:1px solid black;padding:10px;text-align:left"> Top {} Predictions </th>
        '''.format(inputs[i], true_output[i], K)
        for k in range(K):
            html_str += '''
            <td style="color:#000;background-color:{};border:1px solid black;padding:10px"> {} </td>
            '''.format(get_clr(pred_scores[i][k]/5,cmap), pred_output[i][k])
        html_str += '''
        </tr>
        <tr>
        <th scope="row" style="border:1px solid black;padding:10px;text-align:left"> Levenshtein distance (to true output) &emsp; </th>
        '''
        for k in range(K):
            html_str += '''
            <td style="border:1px solid black;padding:10px"> {} </td>
            '''.format(pred_scores[i][k])
        html_str += '''
        </tr>
        </table>
        '''
        html_body+=html_str+'<br>'
        display(html_print(html_str))
    title = 'with attention' if attention else ''
    html_prefix = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Predictions {}</title>
        </head>
        <body>
        <h4>{}</h4>
        '''.format(title,headline)
    html_suffix = '''
        </body>
        </html>
        '''
    html_out = html_prefix+html_body+html_suffix
    fname = "predictions_attn.html" if attention else "predictions.html"
    Func = open(fname,"w")
    Func.write(html_out)
    Func.close()
    if wandb_log:
        run = wandb.init(project="cs6910_assignment3_viz", entity="dlstack", reinit=True)
        wandb.run.name = 'best-model-attn-predictions' if attention else 'best-model-predictions'
        wandb.log({f'Predictions {title}':wandb.Html(open(fname))})
        run.finish()
    print('\n\n')
    
    return random_seq

#Best run id of wandb without attention
best_run_id = "nrd2ctiz"
# acc, exact_B_acc, exact_acc, loss, true_out, pred_out, pred_scores,model = test_model(best_run_id,test_encoder_input,test_decoder_target,max_decoder_seq_length,target_token_index,reverse_target_char_index,save_pred=True,test_input_texts=test_input_texts)
acc, exact_B_acc, exact_acc, loss, true_out, pred_out, pred_scores,model = test_model(best_run_id,test_encoder_input,test_decoder_target,max_decoder_seq_length,target_token_index,reverse_target_char_index,save_pred=False,test_input_texts=test_input_texts)

# random_samples = print_samples(test_input_texts,true_out,pred_out,pred_scores,wandb_log=True)
random_samples = print_samples(test_input_texts,true_out,pred_out,pred_scores,wandb_log=False)

#For plotting model without attention using keras.utils
plot_model(model,show_shapes=True,to_file='best_model.png')

#Best run id of wandb with attention
best_attn_run_id = "nkg32le7"
# acc_a, exact_B_acc_a, exact_acc_a, loss_a, true_out_a, pred_out_a, pred_scores_a,attention_scores,model_a = test_model(best_attn_run_id,test_encoder_input,test_decoder_target,max_decoder_seq_length,target_token_index,reverse_target_char_index,attention=True,save_pred=True,test_input_texts=test_input_texts)
acc_a, exact_B_acc_a, exact_acc_a, loss_a, true_out_a, pred_out_a, pred_scores_a,attention_scores,model_a = test_model(best_attn_run_id,test_encoder_input,test_decoder_target,max_decoder_seq_length,target_token_index,reverse_target_char_index,attention=True,save_pred=False,test_input_texts=test_input_texts)

# random_samples_attn = print_samples(test_input_texts,true_out_a,pred_out_a,pred_scores_a,attention=True,wandb_log=True,random_seq=random_samples)
random_samples_attn = print_samples(test_input_texts,true_out_a,pred_out_a,pred_scores_a,attention=True,wandb_log=False,random_seq=random_samples)

#For plotting model with attention using keras.utils
plot_model(model_a,show_shapes=True,to_file='best_model_attn.png')

#Download font so that matplotlib can display bengali characters
filename = 'Hind_Siliguri'
url = 'https://fonts.google.com/download?family=Hind%20Siliguri'
if not os.path.exists(filename+'.zip') and not os.path.exists(filename):
    filename_zip = wget.download(url)
    with ZipFile(filename_zip, 'r') as z:
        z.printdir()
        print('\nExtracting files ....')
        z.extractall()
        print('Done')
    os.remove(filename_zip)
elif not os.path.exists(filename):
    filename_zip = filename + '.zip'
    with ZipFile(filename_zip, 'r') as z:
        z.printdir()
        print('\nExtracting files ....')
        z.extractall()
        print('Done')
    os.remove(filename_zip)

def plot_heatmaps(inputs,pred_out,pred_scores,attn_scores,wandb_log=False,random_seq=None,cmap='magma'):
    '''
    Function to generate attention heatmaps for 9 samples in a 3 x 3 grid
    Arguments :
        inputs -- input words
        pred_out -- B predicted output words
        pred_scores -- levenshtein distance for the predictions to the true output
        attn_scores -- attention scores
        wandb_log -- (bool, default : False) whether or not to log the image generated to WANDB
        rand_seq -- list of indices from the dataset passed for which the sample outputs are to be printed (If None, random 9 samples will be chosen)
                    (The length of list passed should be >= 9)
    Returns :
        rand_seq -- the list of indices for which sample outputs are printed
    '''
    num_samples = len(pred_out)
    if random_seq is None:
        random_seq = random.sample(range(num_samples),9)
    random_seq = random_seq[:9]
    
    plt.close('all')
    fig = plt.figure(figsize=(15,15))
    fig,axes = plt.subplots(3,3, figsize=(15, 15),constrained_layout=True)
    plt.suptitle('Attention Heatmaps',fontsize='x-large')
    for i,ax in zip(random_seq,axes.flat):
        K = len(pred_scores[i])
        k = np.argmin(pred_scores[i])
        X = attn_scores[i,k,:len(pred_out[i][k])+1,:len(inputs[i])+1].T
        im = ax.imshow(X,vmin=0,vmax=1,cmap=cmap)
        ax.set_xticks(range(len(pred_out[i][k])+1))
        ax.set_xticklabels(list(pred_out[i][k])+['<end>'],fontproperties=FontProperties(fname='HindSiliguri-Medium.ttf'))
        ax.set_yticks(range(len(inputs[i])+1))
        ax.set_yticklabels(list(inputs[i])+['<end>'])
        ax.set_ylabel(f'Encoder Input:{inputs[i]}')
        ax.set_xlabel(f'Decoder Output:{pred_out[i][k]}',fontproperties=FontProperties(fname='HindSiliguri-Medium.ttf'))
        ax.set_title(str(i) + r'$^{th}$ example of Test Set')
        ax.set_aspect("equal")
        ax.grid(False)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.7)
    if wandb_log:
        run = wandb.init(project="cs6910_assignment3_viz", entity="dlstack", reinit=True)
        wandb.run.name = 'attn_heat_maps'
        wandb.log({'attention_heatmaps':fig})
        run.finish()
    plt.show()
    return random_seq
_ = plot_heatmaps(test_input_texts,pred_out_a,pred_scores_a,attention_scores,wandb_log=False,random_seq=random_samples_attn)
# _ = plot_heatmaps(test_input_texts,pred_out_a,pred_scores_a,attention_scores,wandb_log=True,random_seq=random_samples_attn)



def cstr(s, color=None):
      # Function to get text html element
    if color is None:
        return '''<text style="padding:2px; color:#C0C0C0"> {} </text>'''.format(s)
    return '''<text style="color:#000;background-color:{}; padding:2px; color:#FF6699"> {} </text>'''.format(color, s)


def print_connectivity(inputs, pred_out, pred_scores, attn_scores, dec_char_ind=0):
    '''
    Function to visualize attention for one index of decoder output of one sample
    Arguments :
        input -- sample input word
        pred_out -- K predicted output words for the sample
        pred_scores -- levenshtein distance for the predictions to the true output
        attn_scores -- attention scores
        dec_char_ind -- (default : 0) index of the character in decoder for which the visuzalization is to be done
    Returns :
        -- None --
    '''
    K = len(pred_scores)
    print('-'*20 + f' Visualizing attention for Top {K} predictions (in decreasing order of probabilities) ' + '-'*20)
    print('')
    html_str = '''
    <table style="border:2px solid black; border-collapse:collapse; font-size:1.5em">
    <caption> <strong>INPUT : </strong> {} </caption>
    <tr>
    <th style="border:1px solid black;padding:10px;text-align:center"> Character in Prediction Focussed </th>
    <th style="border:1px solid black;padding:10px;text-align:center"> Attention Visualization </th>
    </tr>
    '''.format(inputs)
    for k in range(K):  
        char = pred_out[k][dec_char_ind] if dec_char_ind < len(pred_out[k]) else '&lt end &gt' if dec_char_ind == len(pred_out[k]) else '&lt blank &gt'
        middle_char = pred_out[k][dec_char_ind] if dec_char_ind < len(pred_out[k]) else ''
        end_str = pred_out[k][dec_char_ind+1:] if dec_char_ind < len(pred_out[k])-1 else ''
        html_str += '''
        <tr>
        <td style="border:1px solid black;padding:10px;text-align:center"> character at index {} of {}<span style="color: #FF1493">{}</span>{} <br/> ({}) </td>
        <td style="border:1px solid black;padding:10px;text-align:center">
        '''.format(dec_char_ind, pred_out[k][:dec_char_ind], middle_char, end_str, char)
        for i,c in enumerate(inputs):
            html_str += '''
            {}
            '''.format(cstr(c, get_clr(attn_scores[k,dec_char_ind,i], 'YlGnBu')))
        html_str += '''
        </td>
        </tr>
        '''
    html_str += '''
    </table>
    '''
    display(html_print(html_str))

def visualize_attention(sample_ind=0, dec_char_ind=0):
    # Function to visualize the importance of encoder input characters to the (dec_char_ind)th character of the output,
    # for the (sample_ind)th sample in the test data
    print_connectivity(test_input_texts[sample_ind], pred_out_a[sample_ind], pred_scores_a[sample_ind], attention_scores[sample_ind], dec_char_ind)

# Question 6 - visualizing attention
@interact(sample_ind = IntSlider(min=0, max=len(test_input_texts)-1, step=1, value=10, layout=Layout(width='800px')))
def f(sample_ind):
    print(f'Input : {test_input_texts[sample_ind]}')
    print(f'Top {len(pred_out_a[sample_ind])} predictions : ')
    mx_len = 0
    for pred in pred_out_a[sample_ind]:
        print(pred)
        mx_len = max(mx_len, len(pred))
    
    @interact(character_ind = IntSlider(min=0, max=mx_len-1, step=1, value=0))
    def g(character_ind):
        visualize_attention(sample_ind, character_ind)

