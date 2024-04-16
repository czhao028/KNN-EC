import os
import json
import torch
import pickle
import pandas as pd
from torch.utils.data import TensorDataset
import re
import csv
# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
# Construct the path to the JSON file
json_file_path = os.path.join(parent_directory, 'data', 'emotionToId.json')

# Read the JSON file
with open(json_file_path, 'r') as file:
    emotionNum2ekmanNum = json.load(file)

#Same as from https://github.com/cl-victor1/ling573-project/blob/main/src/agentic_workflow/fine_tune.py
def replace_repeated_punctuation(text): # NB: this function might have adverse impact
    # Define a regular expression pattern to match repeated punctuation separated by spaces
    pattern = r'(\s*[\.,;:!?]+\s*)+'
    # Replace occurrences of the pattern with a single punctuation mark
    replaced_text = re.sub(pattern, lambda m: m.group(1)[0] + " ", text)
    return replaced_text

def merge_capitalized_letters(input_string):
    def replace_match(match):
        # Get the matched substring
        matched_string = match.group(0)
        # Compress uppercase letters separated by spaces
        replacement_string = re.sub(" ", "", matched_string)
        return replacement_string
    # Define the pattern
    pattern = r'\b[A-Z](?:\s[A-Z])*\b'  # Pattern to match a string with uppercase letters separated by spaces
    return re.sub(pattern, replace_match, input_string)

def clean_tweet(tweet):
    tweet = tweet.replace("@user", "")
    tweet = tweet.replace("http", "")
    tweet = replace_repeated_punctuation(tweet)
    tweet = merge_capitalized_letters(tweet)
    return tweet


def getData(tokenizer,file_name):
    
    data=pd.read_csv(file_name,sep='\t')
    
    #data=data[data[1]!='27'] #Remove neutral labels
    #data=data[[len(label.split(','))==1 for label in data[1].tolist()]] #Remove mutil labels
    
    sents=[tokenizer(clean_tweet(sent.lower()),padding='max_length',truncation=True,max_length=128) for sent in data.iloc[:, 1].values.tolist()]
    sents_input_ids=torch.tensor([temp["input_ids"] for temp in sents])
    sents_attn_masks=torch.tensor([temp["attention_mask"] for temp in sents])
    labels=torch.tensor([emotionNum2ekmanNum[label] for label in data.iloc[:, 2].values.tolist()])
    dataset=TensorDataset(sents_input_ids,sents_attn_masks,labels)
    
    return dataset

def getTrainData(tokenizer,bert_name,data_path):
    if not os.path.exists(data_path+ "/%s"%(bert_name.split('/')[-1])):
        os.makedirs( data_path+"/%s"%(bert_name.split('/')[-1]))
    
    feature_file = data_path+"/%s/train_features.pkl"%(bert_name.split('/')[-1])
    if os.path.exists(feature_file):
        train_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        train_dataset = getData(tokenizer,data_path+'/train.tsv')
        with open(feature_file, 'wb') as w:
            pickle.dump(train_dataset, w)
    return train_dataset

def getDevData(tokenizer,bert_name,data_path):
    feature_file = data_path+"/%s/dev_features.pkl"%(bert_name.split('/')[-1])
    if os.path.exists(feature_file):
        dev_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        dev_dataset = getData(tokenizer,data_path+'/dev.tsv')
        with open(feature_file, 'wb') as w:
            pickle.dump(dev_dataset, w)
    return dev_dataset


def getTestData(tokenizer,bert_name,data_path):
    feature_file = data_path+"/%s/test_features.pkl"%(bert_name.split('/')[-1])
    if os.path.exists(feature_file):
        test_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        test_dataset = getData(tokenizer,data_path+'/test.tsv')
        with open(feature_file, 'wb') as w:
            pickle.dump(test_dataset, w)
    return test_dataset

def saveTestResults(test_data_input_filename, prediction_values, output_file_name):
    data = pd.read_csv(test_data_input_filename, sep='\t')
    sentIds = data.iloc[:, 0].values.tolist()
    tsvfile = open(output_file_name, 'w', newline='')
    writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
    writer.writerow(["ID", "Labels"])
    for i, predict_val in enumerate(prediction_values):
        writer.writerow([sentIds[i], predict_val])