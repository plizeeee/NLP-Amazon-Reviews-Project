#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import gzip
import json
from datetime import datetime
import re
import json
import pickle
import random
import numpy as np
import pandas as pd
from html import unescape
from  matplotlib import pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from sklearn.feature_extraction import _stop_words
from collections import defaultdict, Counter
import pickle
import gensim
from gensim.models import FastText
from gensim.test.utils import common_texts  # some example sentences
from sklearn.model_selection import train_test_split
import operator
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from datetime import datetime
from IPython.display import Markdown, display
from gensim.test.utils import common_texts  # some example sentences
from sklearn.model_selection import train_test_split
import operator
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from datetime import datetime
import seaborn as sns


pd.set_option('display.max_colwidth', -1)
pd.options.mode.chained_assignment = None


def train_all_models_and_return_results():
    # In[36]:

    start = datetime.now()
    with open(r'C:\Users\plize\Documents\GitHub\Natural Naguage Processing\Summative\1049140_Natural_Language_Processing_Summative\df_train.p', 'rb') as handle:
        df_train = pickle.load(handle)
        
    with open(r'C:\Users\plize\Documents\GitHub\Natural Naguage Processing\Summative\1049140_Natural_Language_Processing_Summative\df_dev.p', 'rb') as handle:
        df_dev = pickle.load(handle)
        
    with open(r'C:\Users\plize\Documents\GitHub\Natural Naguage Processing\Summative\1049140_Natural_Language_Processing_Summative\df_test.p', 'rb') as handle:
        df_test = pickle.load(handle)
        

    # In[37]:


    # Storing accuracies and confusion matrices for each category of amazon products for displaying all the results of this notebook together later
    review_by_category_accs = dict()
    review_by_category_confusion_matrix = dict()


    # ### Starting by creating models for the Movies and TV section (models will be trained for the Books sections later)

    # In[38]:


    # Starting with 1st category, and will repeat the same analysis with another category later

    category = 'Books'

    df_train_category = df_train[df_train['category']==category].copy()
    df_dev_category = df_dev[df_dev['category']==category].copy()
    df_test_category = df_test[df_test['category']==category].copy()


    # In[39]:


    # Creating default dictionaries for naive bayes


    random_seed = 123

    # Define gender categories
    categories = list(range(6))

    # Initialize data structures
    train_dict = defaultdict(list)
    test_dict= defaultdict(list)

    # put the content of the reviews in a default dict
    for c_i in categories:
        train_dict[c_i] =  df_train_category[df_train_category['class']==c_i].copy()['cleaned_text_lst'].to_list()
        test_dict[c_i] =  df_test_category[df_test_category['class']==c_i].copy()['cleaned_text_lst'].to_list()


    # In[40]:


    # Initialize data structures
    vocab = defaultdict(Counter)
    # n_posts = defaultdict(Counter)

    # Create vocabularies
    for c_i in categories:
        for p in train_dict[c_i]:
            vocab[c_i].update(p)
    #         n_posts[c_i].update(set(p))


    # In[41]:


    # Define function to train Naive Bayes with absolute discounting
    # I modified the code from class to work for multiple classes
    def nb_c(vocab, categories, delta):

        # Calculate number of unseen words for both categories
        vocab_sizes = defaultdict(dict)
        for c_i in categories:
            vocab_sizes[c_i]['seen'] = len(vocab[c_i])
            vocab_sizes[c_i]['unseen'] = 0
            for c_j in categories:
                if c_i == c_j:
                    continue
                vocab_sizes[c_i]['unseen'] += len([w for w in vocab[c_j] if vocab[c_i][w] == 0])

        # Calculate smoothed probabilities
        probs = dict()
        counts = dict()
        for c_i in categories:
            probs[c_i] = {w: vocab[c_i][w] - delta for w in vocab[c_i]}
            for c_j in categories:
                if c_i == c_j:
                    continue
                for w in vocab[c_j]:
                    if vocab[c_i][w] == 0:
                        probs[c_i][w] = delta * (vocab_sizes[c_i]['seen'] / vocab_sizes[c_i]['unseen'])
        
            # Store adjusted counts
            counts[c_i] = probs[c_i]

            total = sum(probs[c_i].values())
            probs[c_i] = {w: probs[c_i][w] / total for w in probs[c_i]}
        return probs, counts


    # In[42]:


    # Perform cross-validation
    # Did not remove most common words as it appears to significantly decrease the accuracy. This makes sense, since the length of the posts
    # has often been cited as an important factor for determining usefulness

    deltas = np.arange(0.9, 0.1, -0.1)
    accs = list()
    # print('Review Summaries: Performing gridsearch with 5-fold cross validation to determine best delta parameter with NB')
    start = datetime.now()

    for d in deltas:
        #start = datetime.now()
        # Shuffle training data
        for c_i in categories:
            shuffle(train_dict[c_i])
        
        # Define total number of train posts and step size
        n = len(train_dict[0])
        s = int(len(train_dict[0]) / 5)
        
        # Initialize list for storing accuracy values
        dev_accs = list()
        
        # Initialize dictionaries for storing adjusted counts
        c_0 = defaultdict(list)
        c_1 = defaultdict(list)
        
        for i in range(0, n, s):
                
            # Initialize training vocabularies
            vocab_train = defaultdict(Counter)
            
            for c_j in categories:
                
                # Loop over cross-validation train posts
                for p in train_dict[c_j][:i] + train_dict[c_j][i+s:]:
                    
                    # Only add content words not in set for removal
                    vocab_train[c_j].update([w for w in p])

            # Train Naive Bayes
            probs, counts = nb_c(vocab_train, categories, d)
            
            # Store adjusted counts
    #         for c_j in categories:
    #             c_0[c_j].append([c for w, c in counts[c_j].items() if vocab_train[c_j][w] == 0][0])
    #             c_1[c_j].append([c for w, c in counts[c_j].items() if vocab_train[c_j][w] == 1][0])
            
            # Initialize lists for storing ground truth labels and predictions
            labels = list()
            preds = list()

            # Loop over gender categories
            for c_i in categories:
                
                # Loop over cross-validation dev posts
                for p in train_dict[c_i][i:i+s]:
                    
                    # Store ground truth
                    labels.append(c_i)
                    
                    # Calculate scores for gender categories
                    scores = {c_j:0 for c_j in categories}
                    for w in p:
                        if w in probs[c_i]:
                            for c_j in categories:
                                scores[c_j] += np.log(probs[c_j][w])
                            
                    # Use higher score for prediction
                    preds.append(max(scores.items(), key=operator.itemgetter(1))[0])
            
            dev_accs.append(len([(l, p)for l, p in zip(labels, preds) if l == p]) / len(labels))
        
    #     # Store mean adjusted counts for delta
    #     for c_i in categories:
    #         c_0_smoothed[c_i].append(np.mean(c_0[c_i]))
    #         c_1_smoothed[c_i].append(np.mean(c_1[c_i]))

        accs.append(np.mean(dev_accs))        
        # print(datetime.now()-start)
        # print('Mean accuracy for delta of {:.2f}: {:.3f}'.format(d, np.mean(dev_accs)))

    d = round(deltas[accs.index(max(accs))],1) # Best delta value
    # print('Best delta value review content from 5 fold cross validation:',d, '   Total elasped time from gridsearch:',datetime.now()-start)
    d_content = d.copy()
    # In[43]:


    # Train on all train posts with best delta

    # Prepare training vocabularies
    vocab_train = defaultdict(Counter)
    for c_i in categories:
        vocab_train[c_i] = Counter({w: vocab[c_i][w] for w in vocab[c_i]})
        
    # Train Naive Bayes
    probs, _ = nb_c(vocab_train, categories, d)

    # Initialize lists for storing ground truth labels and predictions
    labels = list()
    preds = list()

    # Loop over gender categories
    for c_i in categories:
        
        # Loop over test posts
        for p in test_dict[c_i]:
            
            # Store ground truth
            labels.append(c_i)
            
            # Calculate scores for gender categories
    #         scores = {'bad': 0, 'good': 0}
            scores = {c_j:0 for c_j in range(6)}
            for w in p:
                if w in probs[c_i]:
                    for c_j in range(6):
                        scores[c_j] += np.log(probs[c_j][w])
    #                     scores['bad'] += np.log(probs['bad'][w])
            
            # Use higher score for prediction
            preds.append(max(scores.items(), key=operator.itemgetter(1))[0])
                
    #print('Accuracy with best delta on review content alone using overall training data: {:.3f}'.format(len([(l, p)for l, p in zip(labels, preds) if l == p]) / len(labels)))


    # In[44]:


    # Initialize confuction matrix as dictionary
    c_matrix = defaultdict(Counter)

    # Count all training posts
    n = 1000 # 1000 examples per class

    # Create confusion matrix
    for g, p in zip(labels, preds):
        c_matrix[g][p] += 1 / (n)
        
    # pd.DataFrame.from_dict(c_matrix, orient='index', columns=categories).reindex(index=categories)


    # It appears that it is more challenging to seperate 3 star helpful/unhelpful reviews than the other valence rating. It also appears that
    # negative review helpfulness is easier to classify than positive reviews.

    # ### Finding Best Smoothing Parameters for Summaries and years
    # This section is just to find the optimal delta value for the summaries and years, since the value is likely different from the value found
    # for the content of the posts

    # #### Delta for Summary

    # In[45]:


    # Naive bayes for other features (starting with the review summaries)

    # Creating default dictionaries for naive bayes
    random_seed = 123

    # Define gender categories
    categories = list(range(6))

    # Initialize data structures
    train_dict_summaries = defaultdict(list)
    test_dict_summaries= defaultdict(list)

    # put the content of the reviews in a default dict
    for c_i in categories:
        train_dict_summaries[c_i] =  df_train_category[df_train_category['class']==c_i].copy()['summary_cleaned'].to_list()
        test_dict_summaries[c_i] =  df_test_category[df_test_category['class']==c_i].copy()['summary_cleaned'].to_list()
        
    # Create vocabulary list for NB for summaries

    vocab_summaries = defaultdict(Counter)
    # n_posts = defaultdict(Counter)

    # Create vocabularies
    for c_i in categories:
        for p in train_dict_summaries[c_i]:
            vocab_summaries[c_i].update(p)
    #         n_posts[c_i].update(set(p))


    # In[46]:


    # Perform cross-validation to find best delta hyperparameter for summaries
    # Did not remove most common words as it appears to significantly decrease the accuracy. This makes sense, since the length of the posts
    # has often been cited as an important factor for determining usefulness
    deltas = np.arange(0.9, 0.1, -0.1)
    accs = list()
    #print('Review Summaries: Performing gridsearch with 5-fold cross validation to determine best delta parameter with NB')
    start = datetime.now()
    for d in deltas:
        #start = datetime.now()
        # Shuffle training data
        for c_i in categories:
            shuffle(train_dict_summaries[c_i])
        
        # Define total number of train posts and step size
        n = len(train_dict_summaries[0])
        s = int(len(train_dict_summaries[0]) / 5)
        
        # Initialize list for storing accuracy values
        dev_accs = list()
        
        # Initialize dictionaries for storing adjusted counts
        c_0 = defaultdict(list)
        c_1 = defaultdict(list)
        
        for i in range(0, n, s):
                
            # Initialize training vocabularies
            vocab_train_summaries = defaultdict(Counter)
            
            for c_j in categories:
                
                # Loop over cross-validation train posts
                for p in train_dict_summaries[c_j][:i] + train_dict_summaries[c_j][i+s:]:
                    
                    # Only add content words not in set for removal
                    vocab_train_summaries[c_j].update([w for w in p])

            # Train Naive Bayes
            probs, counts = nb_c(vocab_train_summaries, categories, d)
            
            # Store adjusted counts
    #         for c_j in categories:
    #             c_0[c_j].append([c for w, c in counts[c_j].items() if vocab_train[c_j][w] == 0][0])
    #             c_1[c_j].append([c for w, c in counts[c_j].items() if vocab_train[c_j][w] == 1][0])
            
            # Initialize lists for storing ground truth labels and predictions
            labels = list()
            preds = list()

            # Loop over gender categories
            for c_i in categories:
                
                # Loop over cross-validation dev posts
                for p in train_dict_summaries[c_i][i:i+s]:
                    
                    # Store ground truth
                    labels.append(c_i)
                    
                    # Calculate scores for gender categories
                    scores = {c_j:0 for c_j in categories}
                    for w in p:
                        if w in probs[c_i]:
                            for c_j in categories:
                                scores[c_j] += np.log(probs[c_j][w])
                            
                    # Use higher score for prediction
                    preds.append(max(scores.items(), key=operator.itemgetter(1))[0])
            
            dev_accs.append(len([(l, p)for l, p in zip(labels, preds) if l == p]) / len(labels))
        
    #     # Store mean adjusted counts for delta
    #     for c_i in categories:
    #         c_0_smoothed[c_i].append(np.mean(c_0[c_i]))
    #         c_1_smoothed[c_i].append(np.mean(c_1[c_i]))

        accs.append(np.mean(dev_accs))        
        #print(datetime.now()-start)
        #print('Mean accuracy for delta of {:.2f}: {:.3f}'.format(d, np.mean(dev_accs)))


    # #### Delta for Years

    # In[47]:
    d_summary = round(deltas[accs.index(max(accs))],1) # Best delta value
    # print('Best delta value review content from 5 fold cross validation:',d_summary, '   Total elasped time from gridsearch:',datetime.now()-start)

    years_train = defaultdict(dict)
    years_test = defaultdict(dict)


    for c_i in categories:
        years_train[c_i] =  df_train_category[df_train_category['class']==c_i].copy()['review_year_normalized'].to_list()
        years_test[c_i] =  df_test_category[df_test_category['class']==c_i].copy()['review_year_normalized'].to_list()

    # Initialize data structures
    years_occurences_dict = defaultdict(Counter)
    # n_posts = defaultdict(Counter)

    # Create vocabularies
    for c_i in categories:
        for p in years_train[c_i]:
            years_occurences_dict[c_i].update([p])
    #         n_posts[c_i].update(set(p))


    # In[48]:


    # Perform cross-validation to find best delta hyperparameter for summaries
    # Did not remove most common words as it appears to significantly decrease the accuracy. This makes sense, since the length of the posts
    # has often been cited as an important factor for determining usefulness
    accs = list()
    deltas = np.arange(0.9, 0.03, -0.05)
    start = datetime.now()
    for d in deltas:
        # Shuffle training data
        for c_i in categories:
            shuffle(years_train[c_i])
        
        # Define total number of train posts and step size
        n = len(years_train[0])
        s = int(len(years_train[0]) / 5)
        
        # Initialize list for storing accuracy values
        dev_accs = list()
        
        # Initialize dictionaries for storing adjusted counts
        c_0 = defaultdict(list)
        c_1 = defaultdict(list)
        
        for i in range(0, n, s):
                
            # Initialize training vocabularies
            years_occurences_dict = defaultdict(Counter)
            
            for c_j in categories:
                
                # Loop over cross-validation train posts
                for p in years_train[c_j][:i] + years_train[c_j][i+s:]:
                    
                    # Only add content words not in set for removal
                    years_occurences_dict[c_j].update([w for w in [p]])

            # Train Naive Bayes
            probs, counts = nb_c(years_occurences_dict, categories, d)
            
            # Store adjusted counts
    #         for c_j in categories:
    #             c_0[c_j].append([c for w, c in counts[c_j].items() if vocab_train[c_j][w] == 0][0])
    #             c_1[c_j].append([c for w, c in counts[c_j].items() if vocab_train[c_j][w] == 1][0])
            
            # Initialize lists for storing ground truth labels and predictions
            labels = list()
            preds = list()

            # Loop over gender categories
            for c_i in categories:
                
                # Loop over cross-validation dev posts
                for p in years_train[c_i][i:i+s]:
                    
                    # Store ground truth
                    labels.append(c_i)
                    
                    # Calculate scores for gender categories
                    scores = {c_j:0 for c_j in categories}
                    w=p
                    if w in probs[c_i]:
                        for c_j in categories:
                            scores[c_j] += np.log(probs[c_j][w])
                            
                    # Use higher score for prediction
                    preds.append(max(scores.items(), key=operator.itemgetter(1))[0])
            
            dev_accs.append(len([(l, p)for l, p in zip(labels, preds) if l == p]) / len(labels))
        
    #     # Store mean adjusted counts for delta
    #     for c_i in categories:
    #         c_0_smoothed[c_i].append(np.mean(c_0[c_i]))
    #         c_1_smoothed[c_i].append(np.mean(c_1[c_i]))

        accs.append(np.mean(dev_accs))        
    #     print(datetime.now()-start)
        #print('Mean accuracy for delta of {:.2f}: {:.3f}'.format(d, np.mean(dev_accs)))

    d_year = round(deltas[accs.index(max(accs))],1) # Best delta value
    #print('Best delta value review content from 5 fold cross validation:',d_year, '   Total elasped time from gridsearch:',datetime.now()-start)

    # ##### Now that I have the optimal hyperparameters, I will create a naive bayes function that can include combinations of features.
    # The model simply sums the log probabilities (for each included feature) of belonging to a certain class
    # 

    # In[49]:


    # Function for quickly running NB with the best smoothing parameters for each permutation of params
    print('Optimal hyperparameters used for overall NB models:')
    print('d_content = ', d_content,'d_summary = ', d_summary,'d_year = ', d_year)

    def run_nb_best_smoothings(content_bool,summary_bool,years_bool):
        # Note d_content,d_summary and d_year hyperparameters were all obained earlier from the gridsearches
        # d_content = 0.5
        # Prepare training vocabularies for summaries
        vocab_train = defaultdict(Counter)
        for c_i in categories:
            vocab_train[c_i] = Counter({w: vocab[c_i][w] for w in vocab[c_i]})

        # Train Naive Bayes for content of review
        probs_content, _ = nb_c(vocab_train, categories, d_content)


        # d_summary = 0.5

        # Prepare training vocabularies for summaries
        vocab_train_summaries = defaultdict(Counter)
        for c_i in categories:
            vocab_train_summaries[c_i] = Counter({w: vocab_summaries[c_i][w] for w in vocab_summaries[c_i]})

        # Train Naive Bayes for summaries
        probs_summary, _ = nb_c(vocab_train_summaries, categories, d_summary)


        #d_year = 0.65

        # Prepare training vocabularies
        vocab_train_years = defaultdict(Counter)
        for c_i in categories:
            vocab_train_years[c_i] = Counter({w: years_occurences_dict[c_i][w] for w in years_occurences_dict[c_i]})

        # Train Naive Bayes
        probs_years, _ = nb_c(vocab_train_years, categories, d_year)



        # Getting test accuracy

        # Initialize lists for storing ground truth labels and predictions
        labels = list()
        preds = list()

        # Loop over gender categories
        for c_i in categories:

            # Loop over test posts
            for j in range(len(test_dict[c_i])):

                # Going through the content
                p = test_dict[c_i][j]
                # Store ground truth
                labels.append(c_i)

                # Calculate scores for gender categories
        #         scores = {'bad': 0, 'good': 0}
                scores = {c_j:0 for c_j in range(6)}

                # Only use included features
                if content_bool:
                    for w in p:
                        if w in probs_content[c_i]:
                            for c_j in range(6):
                                scores[c_j] += np.log(probs_content[c_j][w])
            #                     scores['bad'] += np.log(probs['bad'][w])

                if summary_bool:
                    # Going through the summaries, and adding the log probabilities
                    p = test_dict_summaries[c_i][j]
                    for w in p:
                        if w in probs_summary[c_i]:
                            for c_j in range(6):
                                scores[c_j] += np.log(probs_summary[c_j][w])
            #               scores['bad'] += np.log(probs['bad'][w])

                if years_bool:
                    # Going through the "years" feature, and adding the log probabilities
                    p = years_test[c_i][j]
                    w = p
                    if w in probs_years[c_i]:
                        for c_j in range(6):
                            scores[c_j] += np.log(probs_years[c_j][w])
            #               scores['bad'] += np.log(probs['bad'][w])


                # Use higher score for prediction
                preds.append(max(scores.items(), key=operator.itemgetter(1))[0])

        acc_val = len([(l, p)for l, p in zip(labels, preds) if l == p]) / len(labels)
    #     print('Accuracy: {:.4f}'.format(acc_val))

        # Initialize confuction matrix as dictionary
        c_matrix = defaultdict(Counter)

        # Count all training posts
        n = 1000 # 1000 examples per class

        # Create confusion matrix
        for g, p in zip(labels, preds):
            c_matrix[g][p] += 1 / (n)

        return [acc_val,c_matrix]


    # In[50]:


    # Combinations of features to permute over while running naive bayes

    # content_bool = True
    # summary_bool = True
    # years_bool = False
    # accs_nb = run_nb_best_smoothings(content_bool,summary_bool,years_bool)

    features_permutations_to_test = {'Content Alone':{'content': True,'summary': False,'years': False},                                 'Non Content':{'content': False,'summary': True,'years': True},                                 'Content and Summary':{'content': True,'summary': True,'years': False},                                 'Content and Year':{'content': True,'summary': False,'years': True},                               'All Features':{'content': True,'summary': True,'years': True}         
                                    }
    # Storing results
    accs_nb = dict()
    conf_mats = dict()
    for run_type in features_permutations_to_test.keys():
        z = features_permutations_to_test[run_type]
        content_bool = features_permutations_to_test[run_type]['content']
        summary_bool = features_permutations_to_test[run_type]['summary']
        years_bool = features_permutations_to_test[run_type]['years']
        outputs_nb = run_nb_best_smoothings(content_bool,summary_bool,years_bool)
        accs_nb[run_type] = outputs_nb[0]
        conf_mats[run_type] = outputs_nb[1]
        


    # In[51]:


    # Storing accuracy of each classifier in a dict (will add accuracies of other models to this dictionnary)
    accs_all_classifiers = dict()
    conf_mats_all_classifiers = dict()
    accs_all_classifiers['nb'] = accs_nb
    conf_mats_all_classifiers['nb'] = conf_mats

    print('\nAccuracies optimal smoothing')
    print(accs_all_classifiers['nb'])


    # The Naive Bayes appears to perform best when all features are used. It also performs better on content alone, than non-content features

    # In[52]:


    # Example confusion matrix after including all features to Naive bayes
    pd.DataFrame.from_dict(conf_mats['All Features'], orient='index', columns=categories).reindex(index=categories)


    # ### Logistic regression and 1-hidden layer neural network models

    # In[53]:

    print('\n\nLogistic Regression TFIDF models')
    # Define logistic regression classifier class
    class LRClassifier(torch.nn.Module):
        
        def __init__(self, input_dim, output_dim):
            
            super(LRClassifier, self).__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)


    # In[54]:


    # Neural network with 1 hidden layer
    class Feed_Forward_Neural_Net(torch.nn.Module):
        
        def __init__(self,input_dim,hidden_layer_size):
            super(Feed_Forward_Neural_Net, self).__init__()
            self.fc1 = torch.nn.Linear(input_dim,hidden_layer_size) # Hidden layer
            self.tanh = torch.nn.Tanh() 
            self.fc2 = torch.nn.Linear(hidden_layer_size,6) # Output layer
            self.softmax = torch.nn.Softmax(dim=1)
            
        def forward(self, x):
            output = self.fc1(x)
            output = self.tanh(output)
            output = self.fc2(output)
            output = self.softmax(output)
            return output


    # In[55]:


    # Function to extract TFIDF vector from train, dev and test set

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Gets the tfidf vectors for the train, dev and test set (requires all 3 at the same time to ensure same transformation is applied)
    def tf_idf_features(train_cleaned_txt,dev_cleaned_txt,test_cleaned_txt, max_feats,max_df):
        tokenized_doc = train_cleaned_txt
        # tokenized_doc = tokenized_doc['cleaned_text_lst'].apply(lambda x: x.split())

        # remove stop-words

        stop_word_lst = list(_stop_words.ENGLISH_STOP_WORDS)
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_word_lst])

        # de-tokenization
        detokenized_doc_train = []
        for i in range(len(tokenized_doc)):
            t = ' '.join(tokenized_doc.iloc[i])
            detokenized_doc_train.append(t)


        # tokenization dev doc
        tokenized_doc = dev_cleaned_txt
        # tokenized_doc = tokenized_doc['cleaned_text_lst'].apply(lambda x: x.split())

        # remove stop-words

        stop_word_lst = list(_stop_words.ENGLISH_STOP_WORDS)
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_word_lst])

        # de-tokenization
        detokenized_doc_dev = []
        for i in range(len(tokenized_doc)):
            t = ' '.join(tokenized_doc.iloc[i])
            detokenized_doc_dev.append(t)



        # tokenization test doc
        tokenized_doc = test_cleaned_txt
        # tokenized_doc = tokenized_doc['cleaned_text_lst'].apply(lambda x: x.split())

        # remove stop-words

        stop_word_lst = list(_stop_words.ENGLISH_STOP_WORDS)
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_word_lst])

        # de-tokenization
        detokenized_doc_test = []
        for i in range(len(tokenized_doc)):
            t = ' '.join(tokenized_doc.iloc[i])
            detokenized_doc_test.append(t)


        vectorizer = TfidfVectorizer(stop_words='english', 
        max_features= max_feats, # keep top 1000 terms 
        max_df = max_df, 
        smooth_idf=True)

        # need to apply same transform to the dev and test sets
        vectorizer.fit(detokenized_doc_train)
        X_train = vectorizer.transform(detokenized_doc_train)
        X_dev = vectorizer.transform(detokenized_doc_dev)
        X_test = vectorizer.transform(detokenized_doc_test)

        return X_train,X_dev,X_test


    # In[56]:


    # Determining the optimal hyperparameters for "max features" and "max_df" by evaluating accuracies on dev set
    param_combination = list()
    acc_params = list()
    for max_feats in(500,1000,2000):
        for min_occurences in(0.3,0.5,0.7):
            
            # Get TFIDF features
            X_train,X_dev,X_test = tf_idf_features(df_train_category['cleaned_text_lst'],                                               df_dev_category['cleaned_text_lst'],                                               df_test_category['cleaned_text_lst'],max_feats,min_occurences) # Getting tfid features
            
            # Training and dev labels
            Y_torch_train = torch.FloatTensor(df_train_category['class'].to_numpy())
            Y_torch_dev = torch.FloatTensor(df_dev_category['class'].to_numpy())
            # Y_torch_dev = Y_torch_test

            # Converting everything to tensors
            X_torch_train_content = torch.FloatTensor(X_train.toarray())
            X_torch_dev_content = torch.FloatTensor(X_dev.toarray())
            # X_torch_dev_content = X_torch_test_content

            # Extracting length of the reviews as an additional feature
            review_length_torch_train = torch.FloatTensor(df_train_category['len_normalized'].to_numpy())
            review_length_torch_dev = torch.FloatTensor(df_dev_category['len_normalized'].to_numpy())
            # review_length_torch_dev = review_length_torch_test
            
            # Concatenating the length of the review to the TFIDF vector
            X_torch_train = torch.cat((X_torch_train_content, review_length_torch_train.unsqueeze(1)), 1)
            X_torch_dev = torch.cat((X_torch_dev_content, review_length_torch_dev.unsqueeze(1)), 1)



            # Creating logistic regression model
            input_dim = X_torch_train.shape[1]
            model = LRClassifier(input_dim,6)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Training the model
            model.train()
            epoch = 401
            loss_val_min = 100
            num_no_progress = 0
            for epoch in range(epoch):
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(X_torch_train)
            #         print(y_pred.shape)
                # Compute Loss
            #     print(min(y_pred))
            #         print(y_pred[1])
                loss = criterion(y_pred.squeeze().float(), Y_torch_train.long())
                with torch.no_grad():
                    y_pred = model(X_torch_dev)
                    correct = 0
                    loss_val = criterion(y_pred.squeeze().float(), Y_torch_dev.long())
                    # if epoch%50 ==0:
                        # print('Epoch {}: train loss: {}       val loss: {}'.format(epoch, loss.item(),loss_val.item()))
                    for i in range(len(Y_torch_dev)):
                        pred = torch.argmax(y_pred[i])
                        ######################### Change this line is loss function changed
                    #             truth = torch.argmax(Y_torch_dev_cv[i])
                        truth = Y_torch_dev[i]
                        if pred == truth:
                            correct += 1
                    acc1 = correct/len(Y_torch_dev)
                    #         acc_vals.append(acc1)
                    if loss_val < loss_val_min:
                        loss_val_min = loss_val
                        num_no_progress = 0
                    else:
                        num_no_progress += 1

                    if num_no_progress >= 5:
                        break
                # Backward pass
                loss.backward()
                optimizer.step()


            # Evalutate accuracy on dev set
            model.eval()
            y_pred = model(X_torch_dev)
            correct = 0
            for i in range(len(Y_torch_dev)):
                pred = torch.argmax(y_pred[i])
                ######################### Change this line is loss function changed
            #             truth = torch.argmax(Y_torch_test_cv[i])
                truth = Y_torch_dev[i]
                if pred == truth:
                    correct += 1
            acc1 = correct/len(Y_torch_dev)


            # print('val accuracy is:',acc1)    
            # print('\n')
            acc_params.append(acc1)
            tfidf_param_dict = dict()
            tfidf_param_dict['max feateatures'] = max_feats
            tfidf_param_dict['min occurences'] = min_occurences
            param_combination.append(tfidf_param_dict)

    best_params_combination_tfidf = param_combination[acc_params.index(max(acc_params))]
    # print(param_combination)
    print('Best tfidf hyperparams')
    print(best_params_combination_tfidf)
    # In[57]:


    # Getting dfidf vectors with optimal hyperparameters
    best_hyperparam_feats = 1000
    best_hyperparam_df = 0.7
    X_train,X_dev,X_test = tf_idf_features(df_train_category['cleaned_text_lst'],                                               df_dev_category['cleaned_text_lst'],                                               df_test_category['cleaned_text_lst'],best_hyperparam_feats,best_hyperparam_df)


    # In[58]:


    # Getting length of summaries and extracting tfidf vector
    df_train_category['len_summary'] = df_train_category['summary_cleaned'].map(lambda x:len(x))
    df_dev_category['len_summary'] = df_dev_category['summary_cleaned'].map(lambda x:len(x))
    df_test_category['len_summary'] = df_test_category['summary_cleaned'].map(lambda x:len(x))

    min_len_summaries = df_train_category['len_summary'].min()
    max_len_summaries = df_train_category['len_summary'].max()

    df_train_category['len_summary_normalized'] = (df_train_category['len_summary']-min_len_summaries)/(max_len_summaries-min_len_summaries)
    df_dev_category['len_summary_normalized'] = (df_dev_category['len_summary']-min_len_summaries)/(max_len_summaries-min_len_summaries)
    df_test_category['len_summary_normalized'] = (df_test_category['len_summary']-min_len_summaries)/(max_len_summaries-min_len_summaries)

    X_train_summary,X_dev_summary,X_test_summary = tf_idf_features(df_train_category['summary_cleaned'],                                               df_dev_category['summary_cleaned'],                                               df_test_category['summary_cleaned'],1000,0.5)


    # In[59]:


    # Relevant tensors (features will be concatenated based on what's included in the model)
    Y_torch_train = torch.FloatTensor(df_train_category['class'].to_numpy())
    Y_torch_dev = torch.FloatTensor(df_dev_category['class'].to_numpy())
    Y_torch_test = torch.FloatTensor(df_test_category['class'].to_numpy())
    # Y_torch_dev = Y_torch_test


    X_torch_train_content = torch.FloatTensor(X_train.toarray())
    X_torch_dev_content = torch.FloatTensor(X_dev.toarray())
    X_torch_test_content = torch.FloatTensor(X_test.toarray())
    # X_torch_dev_content = X_torch_test_content

    X_torch_train_summary = torch.FloatTensor(X_train_summary.toarray())
    X_torch_dev_summary = torch.FloatTensor(X_dev_summary.toarray())
    X_torch_test_summary = torch.FloatTensor(X_test_summary.toarray())
    # X_torch_dev_summary = X_torch_test_summary

    review_length_summary_torch_train = torch.FloatTensor(df_train_category['len_summary_normalized'].to_numpy())
    review_length_summary_torch_dev = torch.FloatTensor(df_dev_category['len_summary_normalized'].to_numpy())
    review_length_summary_torch_test = torch.FloatTensor(df_test_category['len_summary_normalized'].to_numpy())

    X_torch_train_summary = torch.cat((X_torch_train_summary, review_length_summary_torch_train.unsqueeze(1)), 1)
    X_torch_dev_summary = torch.cat((X_torch_dev_summary, review_length_summary_torch_dev.unsqueeze(1)), 1)
    X_torch_test_summary = torch.cat((X_torch_test_summary, review_length_summary_torch_test.unsqueeze(1)), 1)


    review_length_torch_train = torch.FloatTensor(df_train_category['len_normalized'].to_numpy())
    review_length_torch_dev = torch.FloatTensor(df_dev_category['len_normalized'].to_numpy())
    review_length_torch_test = torch.FloatTensor(df_test_category['len_normalized'].to_numpy())
    # review_length_torch_dev = review_length_torch_test

    year_train = torch.FloatTensor(df_train_category['review_year_normalized'].to_numpy())
    year_dev = torch.FloatTensor(df_dev_category['review_year_normalized'].to_numpy())
    year_test = torch.FloatTensor(df_test_category['review_year_normalized'].to_numpy())
    # year_dev = year_test


    # In[61]:


    # Selecting the features used by the model and concatenating the relevant ones
    def included_feats(included_str):
        if included_str == 'All features':
            X_torch_train = torch.cat((X_torch_train_content, review_length_torch_train.unsqueeze(1)), 1)
            X_torch_dev = torch.cat((X_torch_dev_content, review_length_torch_dev.unsqueeze(1)), 1)
            X_torch_test = torch.cat((X_torch_test_content, review_length_torch_test.unsqueeze(1)), 1)

            X_torch_train = torch.cat((X_torch_train, year_train.unsqueeze(1)), 1)
            X_torch_dev = torch.cat((X_torch_dev, year_dev.unsqueeze(1)), 1)
            X_torch_test = torch.cat((X_torch_test, year_test.unsqueeze(1)), 1)

            X_torch_train = torch.cat((X_torch_train, X_torch_train_summary), 1)
            X_torch_dev = torch.cat((X_torch_dev, X_torch_dev_summary), 1)
            X_torch_test = torch.cat((X_torch_test, X_torch_test_summary), 1)

        elif included_str == 'Non content only':
            # Summary of review, and the year
            X_torch_train = torch.cat((X_torch_train_summary, year_train.unsqueeze(1)), 1)
            X_torch_dev = torch.cat((X_torch_dev_summary, year_dev.unsqueeze(1)), 1)
            X_torch_test = torch.cat((X_torch_test_summary, year_test.unsqueeze(1)), 1)

        elif included_str == 'Content only':
            # Features that have to do with the content of the review only (the review embedding, as well as the review length)
            X_torch_train = torch.cat((X_torch_train_content, review_length_torch_train.unsqueeze(1)), 1)
            X_torch_dev = torch.cat((X_torch_dev_content, review_length_torch_dev.unsqueeze(1)), 1)
            X_torch_test = torch.cat((X_torch_test_content, review_length_torch_test.unsqueeze(1)), 1)

        elif included_str == 'Content and summary':
            # Features that have to do with the content of the review only (the review embedding, as well as the review length)
            X_torch_train = torch.cat((X_torch_train_content, review_length_torch_train.unsqueeze(1)), 1)
            X_torch_dev = torch.cat((X_torch_dev_content, review_length_torch_dev.unsqueeze(1)), 1)
            X_torch_test = torch.cat((X_torch_test_content, review_length_torch_test.unsqueeze(1)), 1)

            X_torch_train = torch.cat((X_torch_train, X_torch_train_summary), 1)
            X_torch_dev = torch.cat((X_torch_dev, X_torch_dev_summary), 1)
            X_torch_test = torch.cat((X_torch_test, X_torch_test_summary), 1)

        elif included_str == 'Content and year':
            # Features that have to do with the content of the review only (the review embedding, as well as the review length)
            X_torch_train = torch.cat((X_torch_train_content, review_length_torch_train.unsqueeze(1)), 1)
            X_torch_dev = torch.cat((X_torch_dev_content, review_length_torch_dev.unsqueeze(1)), 1)
            X_torch_test = torch.cat((X_torch_test_content, review_length_torch_test.unsqueeze(1)), 1)

            X_torch_train = torch.cat((X_torch_train, year_train.unsqueeze(1)), 1)
            X_torch_dev = torch.cat((X_torch_dev, year_dev.unsqueeze(1)), 1)
            X_torch_test = torch.cat((X_torch_test, year_test.unsqueeze(1)), 1)
            
        return X_torch_train,X_torch_dev,X_torch_test


    # In[62]:


    review_length_torch_train.shape


    # ### Run the Logistic regression model using the various combinations of features.

    # In[63]:


    # Combinations of features to try
    included_strings = ['Content only','Non content only','Content and summary','Content and year','All features']

    # Store accuracies and confusion matrices
    accs_lr = dict()
    conf_mat_lr = dict()
    
    for included_str in included_strings:
        X_torch_train,X_torch_dev,X_torch_test = included_feats(included_str)
        # print('-----------------------',included_str,'-----------------------')


        # Creating logistic regression model

        input_dim = X_torch_train.shape[1]
        model = LRClassifier(input_dim,6)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Training the model
        model.train()
        epoch = 401
        loss_val_min = 100
        num_no_progress = 0
        for epoch in range(epoch):
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(X_torch_train)
        #         print(y_pred.shape)
            # Compute Loss
        #     print(min(y_pred))
        #         print(y_pred[1])
            loss = criterion(y_pred.squeeze().float(), Y_torch_train.long())
            with torch.no_grad():
                y_pred = model(X_torch_dev)
                correct = 0
                loss_val = criterion(y_pred.squeeze().float(), Y_torch_dev.long())
                #if epoch%50 ==0:
                    # print('Epoch {}: train loss: {}       val loss: {}'.format(epoch, loss.item(),loss_val.item()))
                for i in range(len(Y_torch_dev)):
                    pred = torch.argmax(y_pred[i])
                    ######################### Change this line is loss function changed
                #             truth = torch.argmax(Y_torch_dev_cv[i])
                    truth = Y_torch_dev[i]
                    if pred == truth:
                        correct += 1
                acc1 = correct/len(Y_torch_dev)
                #         acc_vals.append(acc1)
                if loss_val < loss_val_min:
                    loss_val_min = loss_val
                    num_no_progress = 0
                else:
                    num_no_progress += 1

                if num_no_progress >= 5:
                    break
            # Backward pass
            loss.backward()
            optimizer.step()


        # Evalutate accuracy on test set
        model.eval()
        y_pred = model(X_torch_test)
        correct = 0
        for i in range(len(Y_torch_test)):
            pred = torch.argmax(y_pred[i])
            ######################### Change this line is loss function changed
        #             truth = torch.argmax(Y_torch_test_cv[i])
            truth = Y_torch_test[i]
            if pred == truth:
                correct += 1
        acc1 = correct/len(Y_torch_test)

        y_pred = model(X_torch_train)
        correct = 0
        for i in range(len(Y_torch_train)):
            pred = torch.argmax(y_pred[i])
            ######################### Change this line is loss function changed
        #             truth = torch.argmax(Y_torch_test_cv[i])
            truth = Y_torch_train[i]
            if pred == truth:
                correct += 1
        acc2 = correct/len(Y_torch_train)
        accs_lr[included_str] = acc1

        # print('test accuracy is:',acc1)    
        
        nb_classes = 6

        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
        #     for i, (inputs, classes) in enumerate(test_loader):
        #     inputs = inputs.to(device)
        #     classes = classes.to(device)
            outputs = model(X_torch_test)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(Y_torch_test.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        conf_mat_lr[included_str] = confusion_matrix/1000
        # Per class accuracy
        #print('Test set accuracy:')
        #print(torch.mean(confusion_matrix.diag()/confusion_matrix.sum(1)))
        #print('\n')

    # print('\n\n')
    # In[64]:


    # Saving accuracies and confusion matrices for all models to compare later in the analysis
    accs_all_classifiers['lr'] = accs_lr
    conf_mats_all_classifiers['lr'] = conf_mat_lr
    
    # print('\nAccuracies Logistic Regression')
    print('Accuracies LR:',accs_all_classifiers['lr'])
    # ### Run the the Neural network using the various combinations of features.
    
    print('\n\nNN models')

    # Determining the optimal hyperparameters for "max features" and "max_df" by evaluating accuracies on dev set
    param_combination = list()
    acc_params = list()
    for max_feats in(500,1000,2000):
        for min_occurences in(0.3,0.5,0.7):
            
            # Get TFIDF features
            X_train,X_dev,X_test = tf_idf_features(df_train_category['cleaned_text_lst'],                                               df_dev_category['cleaned_text_lst'],                                               df_test_category['cleaned_text_lst'],max_feats,min_occurences) # Getting tfid features
            
            # Training and dev labels
            Y_torch_train = torch.FloatTensor(df_train_category['class'].to_numpy())
            Y_torch_dev = torch.FloatTensor(df_dev_category['class'].to_numpy())
            # Y_torch_dev = Y_torch_test

            # Converting everything to tensors
            X_torch_train_content = torch.FloatTensor(X_train.toarray())
            X_torch_dev_content = torch.FloatTensor(X_dev.toarray())
            # X_torch_dev_content = X_torch_test_content

            # Extracting length of the reviews as an additional feature
            review_length_torch_train = torch.FloatTensor(df_train_category['len_normalized'].to_numpy())
            review_length_torch_dev = torch.FloatTensor(df_dev_category['len_normalized'].to_numpy())
            # review_length_torch_dev = review_length_torch_test
            
            # Concatenating the length of the review to the TFIDF vector
            X_torch_train = torch.cat((X_torch_train_content, review_length_torch_train.unsqueeze(1)), 1)
            X_torch_dev = torch.cat((X_torch_dev_content, review_length_torch_dev.unsqueeze(1)), 1)



            # Creating logistic regression model
            input_dim = X_torch_train.shape[1]
            model = Feed_Forward_Neural_Net(input_dim,500)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Training the model
            model.train()
            epoch = 301
            loss_val_min = 100
            num_no_progress = 500
            for epoch in range(epoch):
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(X_torch_train)
            #         print(y_pred.shape)
                # Compute Loss
            #     print(min(y_pred))
            #         print(y_pred[1])
                loss = criterion(y_pred.squeeze().float(), Y_torch_train.long())
                with torch.no_grad():
                    y_pred = model(X_torch_dev)
                    correct = 0
                    loss_val = criterion(y_pred.squeeze().float(), Y_torch_dev.long())
                    # if epoch%50 ==0:
                        # print('Epoch {}: train loss: {}       val loss: {}'.format(epoch, loss.item(),loss_val.item()))
                    for i in range(len(Y_torch_dev)):
                        pred = torch.argmax(y_pred[i])
                        ######################### Change this line is loss function changed
                    #             truth = torch.argmax(Y_torch_dev_cv[i])
                        truth = Y_torch_dev[i]
                        if pred == truth:
                            correct += 1
                    acc1 = correct/len(Y_torch_dev)
                    #         acc_vals.append(acc1)
                    if loss_val < loss_val_min:
                        loss_val_min = loss_val
                        num_no_progress = 0
                    else:
                        num_no_progress += 1

                    if num_no_progress >= 5:
                        break
                # Backward pass
                loss.backward()
                optimizer.step()


            # Evalutate accuracy on dev set
            model.eval()
            y_pred = model(X_torch_dev)
            correct = 0
            for i in range(len(Y_torch_dev)):
                pred = torch.argmax(y_pred[i])
                ######################### Change this line is loss function changed
            #             truth = torch.argmax(Y_torch_test_cv[i])
                truth = Y_torch_dev[i]
                if pred == truth:
                    correct += 1
            acc1 = correct/len(Y_torch_dev)


            # print('val accuracy is:',acc1)    
            # print('\n')
            acc_params.append(acc1)
            tfidf_param_dict = dict()
            tfidf_param_dict['max feateatures'] = max_feats
            tfidf_param_dict['min occurences'] = min_occurences
            param_combination.append(tfidf_param_dict)

    best_params_combination_tfidf = param_combination[acc_params.index(max(acc_params))]
    # print(param_combination)
    print('Best tfidf hyperparams')
    print(best_params_combination_tfidf)

    X_train,X_dev,X_test = tf_idf_features(df_train_category['cleaned_text_lst'],                                               df_dev_category['cleaned_text_lst'],                                               df_test_category['cleaned_text_lst'],best_hyperparam_feats,best_hyperparam_df)
    
    # Converting training data to tensors
    X_torch_train_content = torch.FloatTensor(X_train.toarray())
    X_torch_dev_content = torch.FloatTensor(X_dev.toarray())
    X_torch_test_content = torch.FloatTensor(X_test.toarray())
    
    # In[65]:


    # Creating Neural Network with 1 hidden layer
    accs_NN = dict()
    conf_mat_NN = dict()
    for included_str in included_strings:
       X_torch_train,X_torch_dev,X_torch_test = included_feats(included_str)
       # print('-----------------------',included_str,'-----------------------\n')    

       input_dim = X_torch_train.shape[1]
       model = Feed_Forward_Neural_Net(input_dim,500)
       criterion = torch.nn.CrossEntropyLoss()
       optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = model.to(device)


       # Training the model
       model.train()
       epoch = 301
       loss_val_min = 100
       num_no_progress = 500
       for epoch in range(epoch):
           optimizer.zero_grad()
           # Forward pass
           y_pred = model(X_torch_train)
       #         print(y_pred.shape)
           # Compute Loss
       #     print(min(y_pred))
       #         print(y_pred[1])
           loss = criterion(y_pred.squeeze().float(), Y_torch_train.long())
           with torch.no_grad():
               y_pred = model(X_torch_dev)
               correct = 0
               loss_val = criterion(y_pred.squeeze().float(), Y_torch_dev.long())
               # if epoch % 10 == 0:
                   # print('Epoch {}: train loss: {}       val loss: {}'.format(epoch, loss.item(),loss_val.item()))
               for i in range(len(Y_torch_dev)):
                   pred = torch.argmax(y_pred[i])
                   ######################### Change this line is loss function changed
               #             truth = torch.argmax(Y_torch_test_cv[i])
                   truth = Y_torch_dev[i]
                   if pred == truth:
                       correct += 1
               acc1 = correct/len(Y_torch_dev)
               #         acc_vals.append(acc1)
               if loss_val < loss_val_min:
                   loss_val_min = loss_val
                   num_no_progress = 0
               else:
                   num_no_progress += 1

               if num_no_progress >= 5:
                   break
           # Backward pass
           loss.backward()
           optimizer.step()


       # Evalutate accuracy on test set
       model.eval()
       y_pred = model(X_torch_test)
       correct = 0
       for i in range(len(Y_torch_test)):
           pred = torch.argmax(y_pred[i])
           ######################### Change this line is loss function changed
       #             truth = torch.argmax(Y_torch_test_cv[i])
           truth = Y_torch_test[i]
           if pred == truth:
               correct += 1
       acc1 = correct/len(Y_torch_test)

       y_pred = model(X_torch_train)
       correct = 0
       for i in range(len(Y_torch_train)):
           pred = torch.argmax(y_pred[i])
           ######################### Change this line is loss function changed
       #             truth = torch.argmax(Y_torch_test_cv[i])
           truth = Y_torch_train[i]
           if pred == truth:
               correct += 1
       acc2 = correct/len(Y_torch_train)

       accs_NN[included_str] = acc1

       
       # print('test accuracy is:',acc1)
       
       nb_classes = 6

       confusion_matrix = torch.zeros(nb_classes, nb_classes)
       with torch.no_grad():
       #     for i, (inputs, classes) in enumerate(test_loader):
       #     inputs = inputs.to(device)
       #     classes = classes.to(device)
           outputs = model(X_torch_test)
           _, preds = torch.max(outputs, 1)
           for t, p in zip(Y_torch_test.view(-1), preds.view(-1)):
                   confusion_matrix[t.long(), p.long()] += 1

       conf_mat_NN[included_str] = confusion_matrix/1000
       # Per class accuracy
       # print('Per Class Accuracy:')
       # print(confusion_matrix.diag()/confusion_matrix.sum(1))
       # print('\n\n\n')


    # In[66]:


    # Saving accuracies and confusion matrices for later analysis
    accs_all_classifiers['nn'] = accs_NN
    conf_mats_all_classifiers['nn'] = conf_mat_NN

    review_by_category_accs['Moviestv'] = accs_all_classifiers
    review_by_category_confusion_matrix['Moviestv'] = conf_mats_all_classifiers
    print('Accuracies NN:', accs_all_classifiers['nn'])
    # print(accs_all_classifiers,conf_mats_all_classifiers)

    return accs_all_classifiers,conf_mats_all_classifiers