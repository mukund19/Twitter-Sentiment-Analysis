
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import preprocessor as p

from sklearn import model_selection
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

stop = ["a","an","is","this","the","that","as","or","to","on","me","has",
    "had","would","will","be","at","because","bcoz","beside","besides","between",
    "until","namely","here", "hereafter", "hereby", "herein", "hereupon", "it","its",
    "seem","former","latter", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever","<e>","<a>","</e>","</a>"]


# In[2]:

# Methods For Printing 

def print_df_stat(df):
    print("Head:\n-------")
    print(df.head(25))
    print("-" * 50)
    print("Info:\n-------")
    print(df.info())
    print("-" * 50)
    print("Class Count\n----------")
    print(df["Class"].value_counts(dropna=False))

def print_10Fold_Stats(model_name, sheet_name, arg_acc, arg_prec, arg_recall, arg_fscore):
    print("-" * 10)
    print(sheet_name + " Classifier: " + model_name + " (Averaged over 10-folds)")    
    print("\nAccuracy \t %f" % (arg_acc/10.0))
    for i in xrange(-1,2):
        print_class_prf(i, (arg_prec/10.0), (arg_recall/10.0), (arg_fscore/10.0))        
    
def print_class_prf(sent, arg_prec, arg_recall, arg_fscore):
    print("Class %d \t Precision: %f, Recall: %f, F-Score: %f"%(sent, arg_prec[sent+1], arg_recall[sent+1], arg_fscore[sent+1]))


# In[3]:

# Methods for Reading and Cleaning Tweets

def read_tweets_excelsheet(filepath,sheetname,fetch_col_list=[3,4]):
    tweet_xl = pd.ExcelFile(filepath)
    df = pd.read_excel(tweet_xl, sheetname, parse_cols=[3,4])
    df.drop(df[(df.Class == "irrevelant") | (df.Class == "irrelevant") |(df.Class == "!!!!") | (df.Class == "IR")].index, inplace=True)
    df.dropna(inplace=True)
    df[['Class']] = df[['Class']].apply(pd.to_numeric)
    df.drop(df[(df.Class == 2)].index, inplace=True)
    df = df.rename(columns={"Anootated tweet":"Tweet"})
    df.reset_index(inplace=True,drop=True)
    return df

def cleantweet(tweet):
    t = p.clean(tweet.strip().encode("utf-8"))
    t = t.lower()
    return t


# In[4]:

def get_XY_ndarray_from_df(df, print_features=False):
    #tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 3,stop_words=stop)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2),stop_words=stop)
    tfidf_matrix =  tf.fit_transform(df['Tweet'])
    
    X = tfidf_matrix.todense().view(type=np.ndarray)
    Y = df['Class'].as_matrix()
    
    if print_features:
        feature_names = tf.get_feature_names() 
        print(len(feature_names))
        print(feature_names)
        
    return X,Y, tf


# In[5]:

def model_train_crossval(dX, dY, model, model_name, sheet_name):
    kf = model_selection.StratifiedKFold(n_splits=10)
    acc = 0.0
    prec = np.zeros(3)
    recall = np.zeros(3)
    fscore = np.zeros(3)
    
    for tr_idx, te_idx in kf.split(dX,dY):                                                      
        # train                                                                                     
        model.fit(dX[tr_idx], dY[tr_idx])
        
        # predict                                                                                   
        dYpred = model.predict(dX[te_idx])                                                        
        
        # get the average accuracy                                                     
        acc += np.mean(dY[te_idx] == dYpred)
        
        prf = precision_recall_fscore_support(dY[te_idx], dYpred, average=None, labels=[-1,0,1])
        prec += prf[0]
        recall += prf[1]
        fscore += prf[2]
    
    print_10Fold_Stats(model_name, sheet_name, acc, prec, recall, fscore)
    return acc


# In[6]:

def model_train(dX, dY, model, model_name, sheet_name):
    
    acc = 0.0
    model.fit(dX, dY)
    dYpred = model.predict(dX)
    
    acc = np.mean(dY == dYpred)
    
    prf = precision_recall_fscore_support(dY, dYpred, average=None, labels=[-1,0,1])
        
    '''
    print("Classifier: {} - Training Accuracy:{}".format(model_name, acc))
    for i in xrange(-1,2):
        print_class_prf(i, (prf[0]), (prf[1]), (prf[2]))
    print("-"*20)
    '''
    return acc


# In[7]:

def train(argfilepath, argsheetname):
    df = read_tweets_excelsheet(filepath=argfilepath,sheetname=argsheetname)
    df["Tweet"] = df["Tweet"].apply(cleantweet)
    X, Y, tf = get_XY_ndarray_from_df(df)
    
    classifiers = {}
    
    lr = LogisticRegression()
    #acc = model_train_crossval(X,Y,lr,"Logistic Regression",argsheetname)
    acc = model_train(X,Y,lr,"Logistic Regression",argsheetname)
    classifiers["Logistic Regression"] = (acc,lr)
    
    bnb = BernoulliNB()
    #acc = model_train_crossval(X,Y,bnb,"BernoulliNB",argsheetname)
    acc = model_train(X,Y,bnb,"BernoulliNB",argsheetname)
    classifiers["BernoulliNB"] = (acc,bnb)

    mnb = MultinomialNB()
    #acc = model_train_crossval(X,Y,mnb,"MultinomialNB",argsheetname)
    acc = model_train(X,Y,mnb,"MultinomialNB",argsheetname)
    classifiers["MultinomialNB"] = (acc,mnb)
    
    #rf = RandomForestClassifier()
    ##acc = model_train_crossval(X,Y,rf,"Random Forest",argsheetname)
    #acc = model_train(X,Y,rf,"Random Forest",argsheetname)
    #classifiers["Random Forest"] = (acc,rf)

    #Uncomment it later - Takes a lot of time
    #svm_linear = SVC(kernel="linear")
    #acc = model_train_crossval(X,Y,svm_linear,"SVM - Linear Kernel","Obama") 

    #knn = KNeighborsClassifier(n_neighbors=5)
    #acc = model_train_crossval(X,Y,knn,"k-Nearest Neighbour","Obama")
    
    #ada = AdaBoostClassifier()
    #acc = model_train_crossval(X,Y,ada,"Adaboost","Obama")
    
    return classifiers,tf    


# In[8]:

def test(argfilepath, argsheetname, models, vectorizer):
    #df = read_tweets_excelsheet(filepath=argfilepath,sheetname=argsheetname)
    tweet_xl = pd.ExcelFile(argfilepath)
    df = pd.read_excel(tweet_xl, argsheetname, parse_cols=[0,4])
    #print_df_stat(df)
    if argsheetname == "Obama":
        df.drop(df[(df.Class == "irrevelant") | (df.Class == "irrelevant") |(df.Class == "!!!!") | (df.Class == "IR")| (df.Class == "ir")].index, inplace=True)
    df.dropna(inplace=True)
    df[['Class']] = df[['Class']].apply(pd.to_numeric)
    df.drop(df[(df.Class == 2)].index, inplace=True)
    df = df.rename(columns={"Anootated tweet":"Tweet"})
    df.reset_index(inplace=True,drop=True)
    
    df["Tweet"] = df["Tweet"].apply(cleantweet)
    tfidf_matrix =  vectorizer.transform(df['Tweet'])
    
    Xte = tfidf_matrix.todense().view(type=np.ndarray)
    Yte = df['Class'].as_matrix()
    
    print("Stats for {}".format(argsheetname))
    print("-"*30)
    
    for model_name in models:
        model = models[model_name][1]
        # predict                                                                                   
        Ypred = model.predict(Xte)                                                        
        
        # get the average accuracy                                                     
        acc = np.mean(Yte == Ypred)
        
        prf = precision_recall_fscore_support(Yte, Ypred, average=None, labels=[-1,0,1])
        
        print("Classifier: {} - Testing Accuracy:{}".format(model_name, acc))
        for i in xrange(-1,2):
            print_class_prf(i, (prf[0]), (prf[1]), (prf[2]))
        print("-"*20)
        


# In[9]:

#df_Obama = read_tweets_excelsheet(filepath="training-Obama-Romney-tweets.xlsx",sheetname="Obama")
ob_models, ob_tf = train(argfilepath="training-Obama-Romney-tweets.xlsx", argsheetname="Obama")


# In[10]:

#df_Romney = read_tweets_excelsheet(filepath="training-Obama-Romney-tweets.xlsx",sheetname="Romney")
rom_models, rom_tf = train(argfilepath="training-Obama-Romney-tweets.xlsx", argsheetname="Romney")


# In[11]:

test(argfilepath="testing-Obama-Romney-tweets.xlsx", argsheetname="Obama", models=ob_models, vectorizer=ob_tf)


# In[12]:

test(argfilepath="testing-Obama-Romney-tweets.xlsx", argsheetname="Romney", models=rom_models, vectorizer=rom_tf)


# In[ ]:



