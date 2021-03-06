import random
from nltk.util import ngrams
from nltk.classify import apply_features
import nltk.classify.util
from nltk.classify.api import ClassifierI
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import ConfusionMatrix
from nltk.metrics.scores import accuracy, precision
import numpy as np
import re
import urllib.request
import pandas as pd
import pickle as pkl
#import flask

#https://www.ou.org/benefactor/



def get_jewish_name_data():#->pd.DataFrame
    clean_file = open("jewish_names.txt","w")
    clean_file.close()
    output = open("jewish_names.txt", "a")
    df_jewish_names = pd.DataFrame()

    #regex to extract names from url
    url = urllib.request.urlopen("https://www.ou.org/benefactor/")

    extract_regex = re.compile(r'<li>([A-Za-z \. "\(\)]*?)</li>')

    data = url.read().decode()

    if data:
        found = re.findall(extract_regex, data)
        for name in found:
            #print(name)
            #code modified from https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
            #remove the extra prefix and suffixes from the names
            delim = ' Z"L', 'Z"l', ' A"H', 'Dr. ', 'Drs. ', 'Rabbi ', 'Mr. ', 'Mrs. ', 'Prof. ', 'RABBI ', 'DR. ', 'MRS. ', 'MR. ', 'PROF. ', 'DR. ', 'DRS. '
            pat = '|'.join(map(re.escape, delim))
            without_extras = re.split(pat, name)
            #print(without_extras)
            new_name = "".join(without_extras)#.strip()

            #eliminate dealing with organization and memorial fund dontations
            if "In Memory of" in new_name or "JOURNEY" in new_name or "JEWISH" in new_name or "PHILANTHROPIES" in new_name or "COMPANY" in new_name or "TRAVEL " in new_name or 'IN MEMORY OF' in new_name or 'CAPITAL' in new_name or "Capital" in new_name or "FAMILY" in new_name or "Family" in new_name or "Foundation" in new_name or 'FOUNDATION' in new_name or "in honor of" in new_name or 'IN HONOR OF' in new_name or " Fund" in new_name or ' FUND' in new_name:
                continue

            #get rid of the and
            without_and = re.split(' and |and |AND | AND ', new_name)
          
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
                        output.write(name2 + "\n")
                        new_df = pd.DataFrame([name2])

                    else:

                        # check if the women's maiden name/different last name is present
                        if len(first_name) == 2:
                            name1 = first_name[0].strip() + " " + first_name[1].strip()
                        output.write(name1+ "\n")
                        output.write(name2+ "\n")
                        new_df = pd.DataFrame([name1, name2])

                    df_jewish_names = pd.concat([new_df, df_jewish_names], ignore_index = True)
                    

            elif len(without_and) ==1:
                update = first_name
                if 2<=len(update)<=3:
                    string = update[0].strip()
                    for indx in range(1, len(update)):
                        string += (" "+ update[indx].strip())
                    output.write(string + "\n")
                    new_df = pd.DataFrame([string])
                    df_jewish_names =pd.concat([new_df, df_jewish_names], ignore_index = True)
                    #print(new_df)
        #df_jewish_names['label'] = "JEWISH"

 #       print(df_jewish_names)
    output.close()
    return df_jewish_names

#get_jewish_name_data()
#https://danesheriff.com/Residents
def get_non_jewish_names():
    clean_file = open("non_jewish_names.txt", 'w')
    clean_file.close()
    output = open("non_jewish_names.txt", "a")
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
            string = first.strip() + " "+ last.strip()
            output.write(string+ "\n")
            df= df.append([string], ignore_index = True)

        #df["label"] = "NOT_JEWISH"
    output.close()
    return df
        #print(df)

#get_non_jewish_names()


def extract_features(name, type =3):
    trigrams= list(ngrams(name,3,left_pad_symbol="<s>",right_pad_symbol="<s>"))
    bigrams= list(ngrams(name,2,left_pad_symbol="<s>",right_pad_symbol="<s>"))
    four_grams= list(ngrams(name,4,left_pad_symbol="<s>",right_pad_symbol="<s>"))
    unigrams = list(ngrams(name,1,left_pad_symbol="<s>",right_pad_symbol="<s>"))

    return dict([(ngram, True) for ngram in type])

#this function uses a simpler feature extraction method
#gets the prefix and suffix of first and last name
#of length "type", default is 3
def extract_features_more(name,type):
    name_list = name.split()
    #print(name_list)
    last_suffix = name_list[-1][-type:]
    first_suffix = name_list[0][-type:]
    first_prefix= name_list[0][:type]
    last_prefix = name_list[-1][:type]
    
    return {"last_suffix": last_suffix, "first_suffix": first_suffix, "first_prefix": first_prefix, "last_prefix":last_prefix}

#print(extract_features_more("Esther Werblowsky"))
# process data-
#             a. label it
#             b. random.shuffle() it
def put_together_all_data(type = "trigrams"):

    j_df = get_jewish_name_data()
    nj_df = get_non_jewish_names()
 


    all_names= ([(name[1], "JEWISH") for name in j_df.itertuples()]+[(name[1],"NOT_JEWISH") for name in nj_df.itertuples()] )
    #print(all_names)
    random.shuffle(all_names)

    feature_sets = [(extract_features(name, type), identity) for name, identity in all_names]
    #print(feature_sets)
    size = (len(feature_sets)//5) *4
    train_data, test_data = feature_sets[:size], feature_sets[size:]

    return (train_data,test_data)


#put_together_all_data()

#this is the method that uses the simpler feature extraction
#to create final dataset
def put_all_together_more(type = 3):
    j_df = get_jewish_name_data()
    nj_df = get_non_jewish_names()


    all_names= ([(name[1], "JEWISH") for name in j_df.itertuples()]+[(name[1],"NOT_JEWISH") for name in nj_df.itertuples()] )
    #print(all_names)
    random.shuffle(all_names)

    feature_sets = [(extract_features_more(name, type), identity) for name, identity in all_names]
    #print(feature_sets)
    size = (len(feature_sets)//5) *4
    train_data, test_data = feature_sets[:size], feature_sets[size:]

    return (train_data,test_data)

def train_model(train_data, test_data):
    classifier = NaiveBayesClassifier.train(train_data)
    pred_labels  = [classifier.classify(features) for features, label in test_data]
    accuracy = nltk.classify.util.accuracy(classifier, test_data)
    test_labels = [iden for features, iden in test_data]
    cm = ConfusionMatrix(pred_labels,test_labels)
    prec = precision(set(pred_labels), set(test_labels))
    # print("Accuracy is: ",accuracy)
    # print("Confusion Matrix is", cm)
    # print("Precision is: ", prec )
    return classifier

def main():
    # train_data, test_data = put_together_all_data()
    # #print("Model with the features as trigrams")
    # train_model(train_data,test_data)

    # train_data, test_data = put_together_all_data(type = "bigrams")
    # #print("Model with the features as bigrams")
    # train_model(train_data, test_data)

    # train_data, test_data = put_together_all_data(type = "four_grams")
    # #print("Model with the features as four grams")
    # train_model(train_data, test_data)

    # train_data, test_data = put_together_all_data(type = "unigrams")
    # #print("Model with the features as unigrams")
    # train_model(train_data, test_data)
    
    # new_train, new_test = put_all_together_more()
    # #print("suffix/prefix features set with three")
    # classifier = train_model(new_train, new_test)
    # #classifier.show_most_informative_features()


    # new_train, new_test = put_all_together_more(type =2)
    # print("suffix/prefix features set with two")
    # classifier = train_model(new_train, new_test)
    # classifier.show_most_informative_features()


    new_train, new_test = put_all_together_more(type =4)
    #print("suffix/prefix features set with four")
    classifier = train_model(new_train, new_test)
    # classifier.show_most_informative_features()

    with open('classifier.pkl','wb') as myfile:
        pkl.dump(classifier ,myfile)
    myfile.close()
main()

