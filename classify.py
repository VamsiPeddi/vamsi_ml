import os
import math
from collections import Counter

###################################
# File Author: Vamsi Peddi
# Net ID: vpeddi
# Student ID: 9079650454
# Year and Semester: Spring 2021
####################################


#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding='utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    out_of_vocab = []
    # TODO: add your code here
    with open(filepath,'r', encoding='utf-8') as doc:
        for word in doc:
            word = word.strip()
            if word in vocab and len(word) > 0 and word not in bow:
                bow[word] = 1
            elif word not in vocab:
                out_of_vocab.append(word)
            else :
                bow[word] += 1
    
    bow[None] = len(out_of_vocab)

    return bow


#Helper Function to find number of occurences of a label in a list
def get_label_occurrences(data):
    result = {}
    signs = Counter(k['label'] for k in data if k.get('label'))
    for sign, count in signs.most_common():
        result[sign] = count
    return result

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here    

    #Calling our helper function
    label_count = get_label_occurrences(training_data)

    total_files = 0
    #Loop to calculate total number of files
    for year in label_list:
        total_files = total_files + label_count[year]

    #Loop to calculate probability 
    for year in label_list:
        prob = (label_count[year] + smooth)/(total_files + 2*smooth)
        final_prob = math.log(prob)
        logprob[year] = final_prob
        
    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here
    size_vocab = len(vocab)
    counter = {}
    total_wc = 0
    none_count = 0
    none_prob = 0

    # Looping through training data to to find count of each word
    for label_and_bow in training_data:
        if label_and_bow['label'] == label:
                bow = label_and_bow['bow']
                for i in range(size_vocab):
                    if vocab[i] in bow and vocab[i] not in counter:
                        counter[vocab[i]] = bow[vocab[i]]
                    elif vocab[i] in bow and vocab[i] in counter:
                        counter[vocab[i]] += bow[vocab[i]]
                    elif vocab[i] not in bow and vocab[i] not in counter:
                        counter[vocab[i]] = 0
                
    
    #Finding the total word count and none count
    for label_bow in training_data: 
        if label_bow['label'] == label:
            bow = label_bow['bow']
            total_wc += sum(bow.values())
            none_count += bow[None]

    # Finding probability for each word in vocab
    for i in range(size_vocab):
        prob = (counter[vocab[i]] + smooth)/(total_wc + smooth*(size_vocab+1))
        final_prob = math.log(prob)
        word_prob[vocab[i]] = final_prob

    # Finding None Probability 
    none_prob = (none_count + smooth)/(total_wc + smooth*(size_vocab+1))
    none_prob = math.log(none_prob)
    word_prob[None] = none_prob
    
    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)

    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, label_list)
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')

    return retval


#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    log_prior = model['log prior']

    p_word_2016 = model['log p(w|y=2016)']
    p_word_2020 = model['log p(w|y=2020)']

    prior_2016 = log_prior['2016']
    prior_2020 = log_prior['2020']

    prob_2016 = 0
    prob_2020 = 0

    result=''

    f1 = open(filepath,'r',encoding='utf-8')
    for line in f1:
       for i in line.split():
           if i in p_word_2016:
              prob_2016 = prob_2016 + p_word_2016[i]
           else:
               prob_2016 = prob_2016 + p_word_2016[None]
    f1.close()

    f2 = open(filepath,'r',encoding='utf-8')
    for line in f2:
        for i in line.split():
            if i in p_word_2020:
                prob_2020 = prob_2020 + p_word_2020[i]
            else:
                prob_2020 = prob_2020 + p_word_2020[None]
    f2.close()

    prob_2016 = prob_2016 + prior_2016
    prob_2020 = prob_2020 + prior_2020
    
    if (prob_2020 > prob_2016):
        result='2020'
    else:
        result='2016'

    retval['predicted y'] = result
    retval['log p(y=2016|x)'] =  prob_2016
    retval['log p(y=2020|x)'] = prob_2020

    return retval


