#!/usr/bin/env python
# coding: utf-8

# **Installing dependencies**
# !pip install wget
# !pip install transformers
# !pip install datasets
# **Necessary Imports**

import wget
import os
from zipfile import ZipFile
import random

import pandas as pd
import numpy as np
from IPython.display import display, HTML

def read_data(filename,foldername,url):
    #method to download and extract data from a url and place it in the folder all in the current working directory
    '''
    Arguments:
    filename -- (string) -- filename of downloaded zip file
    url --(string) -- download url from where the zipped file is fetched
    foldername --(string) -- the files from the file are extracted in this folder
    '''
    if not os.path.exists(filename) and not os.path.exists(foldername):
        filename_zip = wget.download(url,filename)
        with ZipFile(filename_zip, 'r') as z:
            z.printdir()
            print('\nExtracting files ....')
            z.extractall(path=foldername)
            print('Done')
        os.remove(filename_zip)
    elif not os.path.exists(foldername):
        with ZipFile(filename, 'r') as z:
            z.printdir()
            print('\nExtracting files ....')
            z.extractall(path=foldername)
            print('Done')
        os.remove(filename)


# ## Two Datasets can be used for fine tuning

filename1 = 'datasets_small.zip'
filename2 = 'datasets_big.zip'
foldername1 = 'datasets_small'
foldername2 = 'datasets_big'
url1 = 'https://storage.googleapis.com/kaggle-data-sets/6776/81739/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220503%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220503T083043Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=55f4aa710ff542d586888c9c86c3fe7fbc30b324eed1971fd3dfb7b44c0da033d64d8db594d368c814bde54174cea54e6e64c0ef8110453f56086bfff28b3e43622c1947e4e44fe2dd87fd421f43c844047c85e7ff5aaf47b77b9505501fe16769ebab99b085f04f7cc9622c6e0f3d3a995a9f10f407279fea01f8fe41b45c492666018074f4b1b9c20877aac198c0a8051193c2f493604591a5dbdb65971cd809f94c9305f405a1c31f291713b78051124ac88b544323c5c05c281dc8d7b3f55ee895b11c0130c1956afd2bec6832ad182f55210a9f282de30788e0d4673a314a703f7d713180c01df763729532c5b933c3381ccf6eb99f914018d0601b3851'
url2 = 'https://storage.googleapis.com/kaggle-data-sets/118366/3314065/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220503%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220503T131835Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=24b3ec919def9e90873f70af803df3847f3c2f4883fe93922440001c7f82613b974c994c942b94e40cadd69c3d807579662b970217cde17a3c35b29061ad5e53f975383fe5c6f1f7b6302245f05b6d94ac1e93a1e7d3e8b3f1cd5f851d98c50c77bb394fc863a3b937276eee68a0a69b34bb637238528b1d5cab8e2b27a3e64bee9d4a9e3882ed1f0d3ca3c7a0b97eef12849e408dee695220d13be4302f87081aebe6a2071c6541f769a204d5c07b68f7aeae84cfaea9215879b21fa4faf4d5ea4bdceba08eb23b644781278527429531c72d41ee82fa8e6bb946614a1482c6f0432369fe4946b4ad1b2936fa59c71d653e037d1ad52512baa26367a31d2cdd'

read_data(filename1,foldername1,url1)
read_data(filename2,foldername2,url2)

lyrics_file = "lyrics-data.csv"

#Only for Dataset 1 this preprocessing is done
rel_data_path = foldername1
data = []
for root,_,files in os.walk(rel_data_path,'r'):
    for file in files:
        file_absolute_path = os.path.abspath(os.path.join(root,file))
    #     print(file_absolute_path)
        with open(file_absolute_path,'r') as f:
#             data.append('<|title|>'+''.join([s.strip() for s in f.readlines()]))
#             data = data + ['<|title|>'+s.strip() for s in f.readlines()]
            data.append(f.read())
#         os.remove(file_absolute_path)
pd.DataFrame(data,columns=['Lyric']).to_csv(foldername1+"/"+lyrics_file,index=False)

lyrics1 = pd.read_csv(foldername1+"/"+lyrics_file)
lyrics2 = pd.read_csv(foldername2+"/"+lyrics_file)

#To get the training data in csv format so as to pass as argument to load_dataset() function
def to_train(lyrics,max_length=1024,truncate=False,max_rows=20000):
    if lyrics == "lyrics1":
        df = lyrics1
        train_file_name = 'train1.csv'
    else:
        df = pd.DataFrame(lyrics2[(lyrics2['language']=='en')]['Lyric']) 
        train_file_name = 'train2.csv'
        df = df[df['Lyric'].apply(lambda x: len(x.split(' ')) < 350 )]
    data = np.array(df['Lyric'].apply(lambda x : x[:max_length] if len(x) > max_length else x))
    if(data.shape[0]>max_rows and truncate):
        train = data[:max_rows]
    else:
        train = data
    pd.DataFrame(train,columns=['lyrics']).to_csv(train_file_name,index=False)
    return train_file_name
    
train_file_name = to_train("lyrics1",truncate=True) #To train with first dataset (small)
# train_file_name = to_train("lyrics2",truncate=True) #To train with second dataset (big)

#Importing Transformers
import transformers

print(transformers.__version__)

#All necessary imports for finetuning
from datasets import ClassLabel
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import TFGPT2LMHeadModel
from transformers import create_optimizer, AdamWeightDecay
from transformers import DefaultDataCollator

import tensorflow as tf

datasets = load_dataset("csv", data_files={"train": train_file_name})

# datasets["train"][10]

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

# show_random_elements(datasets["train"])

'''
To tokenize all our texts with the same vocabulary that was used when training the model, 
we have to download a pretrained tokenizer. This is all done by the `GPT2Tokenizer` class:
'''
model_checkpoint = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint,from_pt=True)

'''
We can now call the tokenizer on all our texts. This is very simple, using the [`map`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map) 
method from the Datasets library. First we define a function that call the tokenizer on our texts:
'''
def tokenize_function(examples):
    return tokenizer(examples["lyrics"])

'''
Then we apply it to all the splits in our `datasets` object, using `batched=True` 
and 4 processes to speed up the preprocessing.
We won't need the `text` column afterward, so we discard it.
'''
tokenized_datasets = datasets.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["lyrics"]
)

'''
If we now look at an element of our datasets, 
we will see the text have been replaced by the `input_ids` the model will need:
Uncomment below to see
'''
# tokenized_datasets["train"][1]

'''
Now for the harder part: we need to concatenate all our texts together then split the result in small chunks of a certain `block_size`. To do this, we will use the `map` method again, with the option `batched=True`. This option actually lets us change the number of examples in the datasets by returning a different number of examples than we got. This way, we can create our new samples from a batch of examples.

First, we grab the maximum length our model was pretrained with. This might be a big too big to fit in your GPU RAM, so here we take a bit less at just 128.
'''

# block_size = tokenizer.model_max_length
block_size = 128

# Then we write the preprocessing function that will group our texts:
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, though you could add padding instead if the model supports it
    # In this, as in all things, we advise you to follow your heart
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

'''
Also note that by default, the `map` method will send a batch of 
1,000 examples to be treated by the preprocessing function. 
So here, we will drop the remainder to make the concatenated tokenized texts a multiple of 
`block_size` every 1,000 examples. You can adjust this behavior by passing a higher batch size (which will also be processed slower). 
You can also speed-up the preprocessing by using multiprocessing:
'''
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

'''
And we can check our datasets have changed: now the samples contain chunks of `block_size` contiguous tokens, 
potentially spanning several of our original texts.
Uncomment below to see the output
'''
# tokenizer.decode(lm_datasets["train"][1]["input_ids"])

model_without_finetune = TFGPT2LMHeadModel.from_pretrained(model_checkpoint)
model = TFGPT2LMHeadModel.from_pretrained(model_checkpoint)

'''
Once we've done that, it's time for our optimizer! 
We can initialize our `AdamWeightDecay` optimizer directly, 
or we can use the `create_optimizer` function to generate an 
`AdamWeightDecay` optimizer with a learning rate schedule. 
In this case, we'll just stick with a constant learning rate for simplicity, so let's just use `AdamWeightDecay`.
This is quite different from the standard Keras way of handling losses, where labels are passed separately and not visible to the main body of the model, and loss is handled by a function that the user passes to `compile()`, which uses the model outputs and the label to compute a loss value.

The approach we take is that if the user does not pass a loss to `compile()`, the model will assume you want the **internal** loss. If you are doing this, you should make sure that the labels column(s) are included in the **input dict** or in the `columns` argument to `to_tf_dataset`.

If you want to use your own loss, that is of course possible too! If you do this, you should make sure your labels column(s) are passed like normal labels, either as the **second argument** to `model.fit()`, or in the `label_cols` argument to `to_tf_dataset`. 
'''
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)

def generate(model,tokenizer,prompt,return_tensor_type='tf'):
    #Method to generate the possible sequences according to the following configurations based on the prompt passed 
    #as input to the function 
    input_ids = tokenizer.encode(prompt,return_tensors=return_tensor_type)
    if return_tensor_type == 'tf':
        generated_text_samples = model.generate(
            input_ids, 
            max_length=500, 
            num_return_sequences=5,
            no_repeat_ngram_size=2,
            repetition_penalty=1.5,
            top_p=0.1,
            temperature=.90,
            do_sample=True,
            top_k=500,
            early_stopping=True
        )
    else:
        generated_text_samples = model.to('cpu').generate(
            input_ids, 
            max_length=500, 
            num_return_sequences=5,
            no_repeat_ngram_size=2,
            repetition_penalty=1.5,
            top_p=0.1,
            temperature=.90,
            do_sample=True,
            top_k=500,
            early_stopping=True
        )
    print("Generated lyrics : ")
    #Print output for each sequence generated above
    for i,sample in enumerate(generated_text_samples):
        print("{}".format(tokenizer.decode(sample, skip_special_tokens=True)))
        print()

'''
Next, we convert our datasets to `tf.data.Dataset`, 
which Keras understands natively. 
`Dataset` objects have a built-in method for this. 
Because all our inputs are the same length, no padding is required, 
so we can use the DefaultDataCollator. 
Note that our data collators are designed to work for multiple frameworks, 
so ensure you set the `return_tensors='tf'` argument to get Tensorflow tensors out - you don't want to accidentally get a load of `torch.Tensor` objects in the middle of your nice TF code!
'''
data_collator = DefaultDataCollator(return_tensors="tf")
train_set = lm_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

history=model.fit(train_set, epochs=5) #Take long time if using dataset_big to train

generate(model,tokenizer,'I love Deep Learning') #Generate output of finetuned model

generate(model_without_finetune,tokenizer,'I love Deep Learning') #Generate output of pretrained model without finetuning

