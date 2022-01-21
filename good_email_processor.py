import pickle as pick
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import re
import pandas as pd
import nltk
import difflib
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

#uncleaned dataset: https://github.com/IBM/nlc-email-phishing/tree/master/data

def save_vect_and_select(vectModel, selectModel, vectName, selectName):

    vect_fileName = "C:\\Users\\hayde\\OneDrive\\Documents\\Final_Project_497\\final_code\\models\\" + vectName  + ".pkl"
    pick.dump(vectModel, open(vect_fileName, 'wb'))
    select_fileName = "C:\\Users\\hayde\\OneDrive\\Documents\\Final_Project_497\\final_code\\models\\" + selectName  + ".pkl"
    pick.dump(selectModel, open(select_fileName, 'wb'))


def word_stemmer(text):

    stem_text = [PorterStemmer().stem(i) for i in text]

    return stem_text


def word_lemmatizer(text):

    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]

    return lem_text


#text cleaning adapted code from Rebecca Vickery on Towards Data Science

def text_cleaning(data_for_clean):


    df = data_for_clean.to_frame()


    df.columns = ['text']


    #normalize text data

    df['text_clean'] = df['text'].str.lower()
    df['text_clean'] = df['text_clean'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    df['text_clean'] = df['text_clean'].apply(lambda elem: re.sub(r"\d+", "", elem))

    # print("\nNormalized:\n" + df.iloc[5, 1])

    #stopword removal

    df['text_clean'] = df['text_clean'].apply(lambda elem: ' '.join([word for word in elem.split() if word not in (stop)]))


    # print("\nStopwords Removed:\n" + df.iloc[5, 1])

    #tokenize
    df['text_clean'] = df['text_clean'].apply(lambda elem: word_tokenize(elem))


    # print("\nTokenized:")

    # print(df.iloc[5, 1])


    # #stemming

    df['text_clean'] = df['text_clean'].apply(lambda elem: word_stemmer(elem))


    # print("\nStemmed:")

    # print(df.iloc[5, 1])


    #lemmatizing

    # df['text_clean'] = df['text_clean'].apply(lambda elem: word_lemmatizer(elem))


    # print("\nLemmatized:")

    # print(df.iloc[5, 1])


    return data_for_clean


#preprocess code adapted from: https://github.com/MahnoorJaved98/Email-Classification

def preprocess(training_label_file, training_data_file):


    #file handling

    labels_file_handler = open(training_label_file, "rb")
    labels = pick.load(labels_file_handler)
    labels_file_handler.close()

    training_data_handler = open(training_data_file, "rb")
    word_data = pick.load(training_data_handler)
    training_data_handler.close()
    word_data = text_cleaning(word_data)


    # test_size is the percentage of events assigned to the test set
    # (remainder go into training)

    features_train, features_test, labels_train, labels_test = train_test_split(
        word_data, labels, test_size=0.1, random_state=42)

    

    # sm = difflib.SequenceMatcher(None, features_test, features_train)

    

    # print("Difference of features raw: ")

    # print(sm.ratio)

    


    # text vectorization

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)


    # feature selection to avoid too many features confusing the model

    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(
        features_train_transformed).toarray()
    features_test_transformed = selector.transform(
        features_test_transformed).toarray()  

    saveVect = "vectorizer"
    saveSelect = "selector"


    save_vect_and_select(vectorizer, selector, saveVect, saveSelect)  
    return features_train_transformed, features_test_transformed, labels_train, labels_test
