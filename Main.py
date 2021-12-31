#scrape data from online- with jewish names
        ##use regex's?
        ##how to make sure not getting organizations
        ##or sort out afterwards
        ##need to seperate names by grabbing each first name of couple and then adding it to last name as seperate entries
#extract_regex = re.compile(r'<li>[DR\.]?[RABBI]?[DRS\.]?([A-Za-z \.]*?)?[A"H]?[Z"L]?([A-Za-z \.]*?)?[A"H]?[Z"L]?</li>')
#extract_regex = re.compile(r'<li>[DR\.|RABBI|DRS\.|MR\.]?([A-Za-z \.]*?)[A"H|Z"L]?[MRS.]?([A-Za-z \.]*?)[A"H|Z"L]?</li>')

##now need to clean data::
# get rid of titles
# get rid of Z"L
# split husband and wife names
# get rid of organizations listed

""""
issues with scraped data:
- three names- one person: eg RUTH BRANDT SPITZER
- What do we do with someone's maiden num + married name
- what to do with this..?STUART KARON AND DR. JODI WENGER
- count for when it only says mr and mrs MALE NAME LAST NAME:eg MR. AND MRS. DAN GOLDISH
-MARILYN RABHAN SWEDARSKY AND DR. ROBERT SWEDARSKY
-Deal with paranthesis-? Mark (Moishe) Bane
"""
"""
delimiters = "a", "...", "(c)"
>>> example = "stackoverflow (c) is awesome... isn't it?"
>>> regexPattern = '|'.join(map(re.escape, delimiters))
>>> regexPattern
'a|\\.\\.\\.|\\(c\\)'
>>> re.split(regexPattern, example)

"""
import random


#https://www.ou.org/benefactor/

import re
import urllib.request
import pandas as pd

def get_jewish_name_data():#->pd.DataFrame

    df_jewish_names = pd.DataFrame()

    #regex to extract names from url
    url = urllib.request.urlopen("https://www.ou.org/benefactor/")

    extract_regex = re.compile(r'<li>([A-Za-z \. "\(\)]*?)</li>')

    data = url.read().decode()

    if data:
        found = re.findall(extract_regex, data)
        for name in found:
            #print(name)
            #remove the extra prefix and suffixes from the names
            delim = ' Z"L', 'Z"l', ' A"H', 'Dr. ', 'Drs. ', 'Rabbi ', 'Mr. ', 'Mrs. ', 'Prof. '
            pat = '|'.join(map(re.escape, delim))
            without_extras = re.split(pat, name)
            #print(without_extras)
            new_name = "".join(without_extras).strip()
            #print(new_name)

            #eliminate dealing with organization and memorial fund dontations
            if "In Memory of" in new_name or "Capital" in new_name or "FAMILY" in new_name or "Family" in new_name or "Foundation" in new_name or "in honor of" in new_name or " Fund" in new_name:
                continue

            #get rid of the and
            without_and = re.split(' and |and ', new_name)
            #print(without_and)
            first_name = without_and[0].split()
            first_name = [first_name[indx][0] + first_name[indx][1:].lower() for indx in range(len(first_name))]
            #now seperate the joint spouse names
            if len(without_and) == 2:
                second_name = without_and[1].split()

                if 2<=len(second_name)<=3:
                    second_name = [second_name[indx][0] + second_name[indx][1:].lower() for indx in range(len(second_name))]

                    if len(second_name) == 3: #allow for initial
                        first, last = second_name[0] +" "+ second_name[1], second_name[2]
                    else:
                        first, last = second_name[0], second_name[1]


                    # assume that its a women's first name with her husband's last name
                    try: name1 = first_name[0].strip() + " " + last.strip()
                    except: name1 = None
                    name2 = first.strip() +" "+ last.strip()

                    # if the wife's first name isn't present
                    #then just output the husbands name to the df
                    if not name1:
                        new_df = pd.DataFrame([name2])

                    else:

                        # check if the women's maiden name/different last name is present
                        if len(first_name) == 2:
                            name1 = first_name[0].strip() + " " + first_name[1].strip()

                        new_df = pd.DataFrame([name1, name2])

                    df_jewish_names = pd.concat([new_df, df_jewish_names], ignore_index = True)
                    #print(new_df)

            elif len(without_and) ==1:
                update = first_name
                if 2<=len(update)<=3:
                    string = update[0].strip()
                    for indx in range(1, len(update)):
                        string += (" "+ update[indx].strip())
                    new_df = pd.DataFrame([string])
                    df_jewish_names =pd.concat([new_df, df_jewish_names], ignore_index = True)
                    #print(new_df)
        #df_jewish_names['label'] = "JEWISH"

 #       print(df_jewish_names)
    return df_jewish_names

#get_jewish_name_data()
#https://danesheriff.com/Residents
def get_non_jewish_names():
    df = pd.DataFrame()

    # regex to extract names from url
    url = urllib.request.urlopen("https://danesheriff.com/Residents")

    extract_regex = re.compile(r'<td>([A-Z a-z,-]*?)</td>')

    data = url.read().decode()

    if data:
        found = re.findall(extract_regex,data)
        #print(found)

        #make the names start with uppercase but the rest lowercase
        #add each name to dataframe, first name and then last name
        for name in found:
            last, first = name.split(',')
            last_split = last.split()
            first_split = first.split()
            cur = 0
            start_with = ""
            while cur< max(len(last_split),len(first_split)):
                if cur>0: start_with = " "
                if cur<len(last_split):
                    last_split[cur] = start_with +last_split[cur][0]+ last_split[cur][1:].lower()
                if cur<len(first_split):
                    first_split[cur] = start_with + first_split[cur][0]+ first_split[cur][1:].lower()
                cur+=1
            last = "".join(last_split)
            first = "".join(first_split)
            df= df.append([first.strip() + " "+ last.strip()])

        #df["label"] = "NOT_JEWISH"
    return df
        #print(df)

#get_non_jewish_names()


from nltk.util import ngrams
from nltk.classify import apply_features
import nltk.classify.util
from nltk.classify.api import ClassifierI
from nltk.classify import NaiveBayesClassifier
import numpy as np

def extract_features(name, type):
    trigrams= list(ngrams(name,3,left_pad_symbol="<s>",right_pad_symbol="<s>"))
    bigrams= list(ngrams(name,2,left_pad_symbol="<s>",right_pad_symbol="<s>"))
    four_grams= list(ngrams(name,4,left_pad_symbol="<s>",right_pad_symbol="<s>"))
    unigrams = list(ngrams(name,1,left_pad_symbol="<s>",right_pad_symbol="<s>"))

    return dict([(ngram, True) for ngram in type])



# process data-
#             a. label it
#             b. random.shuffle() it
def put_together_all_data(type = "trigrams"):

    j_df = get_jewish_name_data()
    nj_df = get_non_jewish_names()
    print(j_df)
    print(nj_df)
    """
    total_jewish = []

    for name in j_df.itertuples():
        total_jewish.append((extract_features(name,type), "JEWISH"))

    total_not_jewish =[]
    for name in j_df.itertuples():
        total_not_jewish.append((extract_features(name, type), "NOT_JEWISH"))

    train_part_j = len(total_jewish)//4 * 3
    train_part_nj = len(total_not_jewish)//4 * 3

    train_data = np.array(total_jewish[:train_part_j]+ total_not_jewish[:train_part_nj])
    test_data = np.array(total_jewish[train_part_j:] + total_not_jewish[train_part_nj:])

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    """

    for name in j_df.itertuples():
        print(name[1])
    all_names= ([(name, "JEWISH") for name in j_df.itertuples()]+[(name,"NOT_JEWISH") for name in nj_df.itertuples()] )


    # #df = pd.DataFrame()
    # df = pd.concat([j_df, nj_df],ignore_index = True)
    # df = df.sample(frac = 1).reset_index(drop=True) #shuffle the dataset
    # print(df)
    return (train_data,test_data)

put_together_all_data()

def train_model(train_data, test_data):
    classifier = NaiveBayesClassifier.train(train_data)
    accuracy = nltk.classify.util.accuracy(classifier, test_data)
    print(accuracy)

def main():
    train_data, test_data = put_together_all_data()
    train_model(train_data,test_data)

    train_data, test_data = put_together_all_data(type = "bigrams")
    train_model(train_data, test_data)

    train_data, test_data = put_together_all_data(type = "four_grams")
    train_model(train_data, test_data)

    train_data, test_data = put_together_all_data(type = "unigrams")
    train_model(train_data, test_data)

main()

"""
        1. create function that returns a dictionary of generated features
                a. can use kitchen sink method- throw features at the model
                     - some to think about, prefix of last name(first three letters), sufix of last name(last three letters), first name,
                     - 
                
        2. process data-
            a. label it
            b. random.shuffle() it
        
        3. create an array of the features of the data using the feature extractor
                - use nltk.classify.apply_features
        4. split into train and test
        5. train a Naive Bayes classifier (nltk? scikit learn?) on training set
 
        
        
"""
#make features from names- ngrams..? trigrams..?

#build classification model that predicts whether a name is jewish or not

