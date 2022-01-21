import pickle as pick
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np
import pandas as pd
from good_email_processor import text_cleaning


model_filename = 'C:\\Users\\hayde\\OneDrive\\Documents\\Final_Project_497\\final_code\\models\\RandomForest_default_params.pkl'

with open(model_filename, 'rb') as f:
    random_forest = pick.load(f)


email_text_ham = ["Subject: ethink about it : november 13 , 2000 have you lost your competitive mind ? find it on the edge . if you ' re looking for competitive intelligence from recent articles , press releases or trends , find it on the edge ! if you have information about recent moves in the market , put it on the edge ! lube stocks trading . . . . maritime weather derivatives . . . viticulturists . no , these aren ' t the results of the ethink team ' s latest word association session . they ' re all ideas in the thinkbank ' s idea vault . visit the thinkbank to get the rest of the story on these ideas . while you ' re there , stop by resources and good sense , too ."]

email_text_spam = ['Subject: unlicensed installation found on your computer get your peace of mind with our affordable softw @ re : order now acapulco globeveldt oligarchy alohagilmore scapula sombreturnery seaside colloqmighty historian trainmenserve never reveriejostle emitter babysatloiter essen braceletbeograd bois javelinpray']



def single_prediction(email_text, model):

    selector_filename = 'C:\\Users\\hayde\\OneDrive\\Documents\\Final_Project_497\\final_code\\models\\selector.pkl'

    vectorizer_filename = 'C:\\Users\\hayde\\OneDrive\\Documents\\Final_Project_497\\final_code\\models\\vectorizer.pkl'

    with open(selector_filename, 'rb') as f:
        selector = pick.load(f)

    with open(vectorizer_filename, 'rb') as f:
        vectorizer = pick.load(f)

    email_text = pd.Series(email_text)
    email_array = text_cleaning(email_text)
    email_array  = vectorizer.transform(email_text)
    email_array_transformed = selector.transform(
        email_array).toarray()
    
    # email_values = email_array_transformed[0]
            
    #print statements that can be useful for debugging
    # print(email_values)
    # print(len(email_values))
    # print(type(email_values))

    # email_values = np.array(email_values)

    array_for_prediction = email_array_transformed.reshape(1, -1)

    pred = model.predict(array_for_prediction)
    pred_proba = model.predict_proba(array_for_prediction)


    return pred_proba


pred_proba = single_prediction(email_text_ham, random_forest)
# print(pred[0])
print(pred_proba[0][0])
# pred, pred_proba = single_prediction(email_text_spam, random_forest)
# print(pred[0])
# print(pred_proba[0][1])
