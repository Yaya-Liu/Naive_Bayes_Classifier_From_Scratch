# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:37:26 2019

@author: Yaya Liu

The goal of this project is to build a Naive Bayes model to identify the attitude (positive/negative) of movie reviews.

This project supports 4 different ways of feature creation: 
    - Stemming reviews + word's frequency based on BOW
    - Non-Stemming reviews + word's frequency based on BOW
    - Stemming reviews + Binary (only keeps whether this word exists in positive or negative training reviews)
    - Non-Stemming reviews + Binary (only keeps whether this word exists in positive or negative training reviews)  
"""

import pandas as pd
import numpy as np
import os, re, glob
import string, nltk
from nltk.tokenize import word_tokenize 
from nltk import PorterStemmer as Stemmer

from sklearn.metrics import classification_report
import logging


def read_files():

    """

    This is a utility function that will read all teh data files
 
    from the data folder and return a dataframe

    the dataframe will have 4 columns:

    test_train: this column has the value test or train

    pos_neg: this column has the value pos_neg

    text: this column has the content of the file
 
    """

    flist = glob.glob('**/*_*.txt', recursive=True)

    test_train_l = []

    pos_neg_l = []

    fname_l = []

    text_l = []

    for fpath in flist:

        parts = fpath.split('\\')

        test_train = parts[2]

        pos_neg = parts[3]

        fname = parts[4]

        text = ""
        
        i = 0 

        with open(fpath, 'rt', encoding = 'utf8') as f:
            if (i <= 10):
                text = f.readlines()
                i += 1

        test_train_l.append(str(test_train))

        pos_neg_l.append(str(pos_neg))

        fname_l.append(fname)

        text_l.append(str(text))

    df = pd.DataFrame({'test_train':test_train_l,

                       'pos_neg':pos_neg_l,

                       'fname':fname,

                       'text':text_l})
    return df

# pre-process reviews
def pre_process(text, para):    
    text = text[2: -2]       # strip [" and "] from the text
    text = re.sub(r'<[^>]+>', "", str(text))  # remove html tags     
    tokens = word_tokenize(text)  
    
    # lower case words if it is not all caps
    words = [word.lower() for word in tokens if word.isupper() == False]      
    
    if (para == "noStem"):
        return words
    else:
        st = Stemmer()
        new_words = [st.stem(t) for t in words]   
        return new_words

# calculate confidential frequency    
def calculate_freq(df, isStem):
    pos_V, neg_V = {}, {}             # frequency dictionary
    count_pos_words, count_neg_words = 0, 0   # number of words in positive review and negative review

    df_train_pos = df[(df['pos_neg'] == 'pos') & (df['test_train'] == 'train')]       
    df_train_neg = df[(df['pos_neg'] == 'neg') & (df['test_train'] == 'train')]
    
    prior_pos = len(df_train_pos)/(len(df_train_pos) + len(df_train_neg))
    prior_neg = len(df_train_neg)/(len(df_train_pos) + len(df_train_neg))
    
    if(isStem == "Stem"):  
        for tokens in df_train_pos['text_clean_Stem']:
            for token in tokens:     
                count_pos_words += 1
                if token not in pos_V.keys():
                    pos_V[token] = 1
                else:
                    pos_V[token] += 1   
        for tokens in df_train_neg['text_clean_Stem']:
            for token in tokens:
                count_neg_words += 1                
                if token not in neg_V.keys():
                    neg_V[token] = 1
                else:
                    neg_V[token] += 1  
    elif(isStem == "noStem"):   
        for tokens in df_train_pos['text_clean_noStem']:            
            for token in tokens:
                count_pos_words += 1                
                if token not in pos_V.keys():
                    pos_V[token] = 1
                else:
                    pos_V[token] += 1                            
        for tokens in df_train_neg['text_clean_noStem']:
            for token in tokens:
                count_neg_words += 1                
                if token not in neg_V.keys():
                    neg_V[token] = 1
                else:
                    neg_V[token] += 1   
    return prior_pos, prior_neg, pos_V, neg_V, count_pos_words, count_neg_words
    
 
# Naive Bayes Model    
def NaiveBayes_model(df, isStem, isFreq):    
    pos_prob_dict, neg_prob_dict = {}, {}       # conditional probability dict   
    preds = []
            
    prior_pos, prior_neg, pos_V, neg_V, count_pos_words, count_neg_words = calculate_freq(df, isStem)
    print("prior_pos:", prior_pos, "prior_neg:", prior_neg)
    print("pos_V:", len(pos_V), "neg_V:", len(neg_V))
    print("count_pos_words:", count_pos_words, "count_neg_words:", count_neg_words,)    

    for key, value in pos_V.items():
        pos_prob_dict[key] = np.float64(np.log((value + 1) / (count_pos_words + len(pos_V) + len(neg_V))))              
    for key, value in neg_V.items():
        neg_prob_dict[key] = np.float64(np.log((value + 1) / (count_neg_words + len(pos_V) + len(neg_V))))           

    df_test = df[df['test_train'] == 'test']
    
    
    if(isFreq == "Frequency"):
        if(isStem == "Stem"):
            for tokens in df_test['text_clean_Stem']:
                pos_p, neg_p = 0, 0
                for token in tokens:
                    if token in pos_prob_dict.keys() and token in neg_prob_dict.keys():
                        pos_p += pos_prob_dict[token]
                        neg_p += neg_prob_dict[token]
                        continue
                    elif token in pos_prob_dict.keys():
                        pos_p += pos_prob_dict[token]
                        neg_p += np.float64(np.log(1/(count_neg_words + len(pos_V) + len(neg_V))))  
                    elif token in neg_prob_dict.keys():
                        pos_p += np.float64(np.log(1/(count_pos_words + len(pos_V) + len(neg_V)))) 
                        neg_p += neg_prob_dict[token]
                    else:
                        continue
                #print(pos_p + np.log(prior_pos), neg_p + np.log(prior_neg))
                if (pos_p + np.log(prior_pos)) > (neg_p + np.log(prior_neg)):
                    preds.append("pos")
                else:
                    preds.append("neg")
            #print(preds)
            
        elif(isStem == "noStem"):
            for tokens in df_test['text_clean_noStem']:
                pos_p, neg_p = 0, 0
                for token in tokens:
                    if token in pos_prob_dict.keys() and token in neg_prob_dict.keys():
                        pos_p += pos_prob_dict[token]
                        neg_p += neg_prob_dict[token]
                        continue
                    elif token in pos_prob_dict.keys():
                        pos_p += pos_prob_dict[token]
                        neg_p += np.float64(np.log(1/(count_neg_words + len(pos_V) + len(neg_V))))  
                    elif token in neg_prob_dict.keys():
                        pos_p += np.float64(np.log(1/(count_pos_words + len(pos_V) + len(neg_V)))) 
                        neg_p += neg_prob_dict[token]
                    else:
                        continue
                if (pos_p + np.log(prior_pos)) > (neg_p + np.log(prior_neg)):
                    preds.append("pos")
                else:
                    preds.append("neg")
            #print(preds)
                
    elif(isFreq == "Binary"):    
        if(isStem == "Stem"):
            for tokens in df_test['text_clean_Stem']:
                pos_num, neg_num = 0, 0
                #tokens = set(tokens)
                for token in tokens:
                    if token in pos_prob_dict.keys() and token in neg_prob_dict.keys():
                        #pos_num += 1
                        #neg_num += 1
                        continue
                    elif token in pos_prob_dict.keys():
                        pos_num += 1
                    elif token in neg_prob_dict.keys():
                        neg_num += 1
                    else:
                        continue  
                if pos_num > neg_num:
                    preds.append("pos")
                else:
                    preds.append("neg")
            #print(preds)
        elif(isStem == "noStem"):
            for tokens in df_test['text_clean_noStem']:
                pos_num, neg_num = 0, 0 
                #tokens = set(tokens)
                for token in tokens:
                    if token in pos_prob_dict.keys() and token in neg_prob_dict.keys():
                        #pos_num += 1
                        #neg_num += 1
                        continue
                    elif token in pos_prob_dict.keys():
                        pos_num += 1
                    elif token in neg_prob_dict.keys():
                        neg_num += 1
                    else:
                        continue            
                if pos_num > neg_num:
                    preds.append("pos")
                else:
                    preds.append("neg")
            #print(preds)
            


    print("=========== Prediction Report: ", isFreq, isStem)
    print("----------- Confusion Matrix -----------")
    print(pd.crosstab(df_test['pos_neg'], pd.Series(preds), rownames = ['True'], colnames = ['Predicted'], margins = True))       
    
    print("----------- Classification Report -----------")
    print(classification_report(df_test['pos_neg'], pd.Series(preds))) 
    

    log = "NB-log.txt"   # create a log file
    logging.basicConfig(filename=log,level = logging.DEBUG, format = '%(message)s')
    logging.info('%script start logging')   # start logging
    
    logging.info("=========== Prediction Report: %s %s", isFreq, isStem)   
    logging.info("----------- Confusion Matrix -----------")   
    logging.info(pd.crosstab(df_test['pos_neg'], pd.Series(preds), rownames = ['True'], colnames = ['Predicted'], margins = True)) 
    
    logging.info("----------- Classification Report -----------")  
    logging.info(classification_report(df_test['pos_neg'], pd.Series(preds))) 
        
    
                            
def main():
    df = read_files()   

    df['text_clean_noStem'] = df['text'].apply(pre_process, args = ("noStem",))
    df['text_clean_Stem'] = df['text'].apply(pre_process, args = ("Stem",))

    
    # Stem/noStem, Freq/Binary
    NaiveBayes_model(df, "Stem", "Frequency")
    NaiveBayes_model(df, "noStem", "Frequency")    
    NaiveBayes_model(df, "Stem", "Binary")      
    NaiveBayes_model(df, "noStem", "Binary") 

if __name__ == "__main__":
    main()
    
