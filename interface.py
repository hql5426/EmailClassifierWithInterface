import gradio as gr
import pickle as pick
from sklearn.feature_extraction.text import TfidfVectorizer
from single_prediction import single_prediction



def id_email_type(email):

    model_filename = 'C:\\Users\\hayde\\OneDrive\\Documents\\Final_Project_497\\final_code\\models\\RandomForest_default_params.pkl'

    with open(model_filename, 'rb') as f:
        model = pick.load(f)
    
    pred = single_prediction(email, model)

    # mText = ""

    # if pred[0] == "ham":
    #     if 0.799 >= pred[0][0] > 0.649:
    #         mText = "I am pretty sure that one is okay!"
    #     elif pred[0][0] > 0.799:
    #         mText = "That's not spam! Definitely don't delete it!!"
    #     elif pred[0][0] < 0.649:
    #         mText = "I don't think that is spam... but I would read it carefully."


    # elif pred[0] == "spam":
    #     if 0.799 >= pred[0][1] > 0.649:
    #         mText = "I'm pretty sure its spam, I would be careful with it!"
            
    #     elif pred[0][1] > 0.799:
    #         mText = "Don't respond!!!! I think it's spam!!!"

    #     elif pred[0][1] < 0.649:
    #         mText = "Hmmm... I would be careful here, this email seems suspicious..."
    
    # else:
    #     mText = "I am not really sure what to make of that!"

    return{'Not Spam' : pred[0][0], 'Spam' : pred[0][1]}

examples = [["Subject: continued customer service commitment the resolution center is taking the next step of ensuring we are providing the service you , our customer , are looking for . as of wednesday september 20 th , with each call made to the resolution center , an email survey will be sent out upon closure of the ticket . please take the time to fill out the survey and let us know how we are doing . we will post these results quarterly on our web page and show exactly how well we are doing . we have attached a copy of the survey for your viewing . thank you again for your continued support . resolution center region : houston status : 4 summary : this is a test your it support case number hdo 000000001606 has now been closed by demo ( demo user for startup ) . in order to improve the quality of support we provide , we would like to know if you were satisfied with the service you received on this call . please select a grade from the list below which best describes the service you received :"], ["Subject: transportation contract # 25374 michelle , please ammend oneok buston processings transportation contract # 25374 to include the month of january , 2001 . thank you , andrew pacheco"], ["Subject: clean ur computer now 3 ey there is a 85 % chance that your computer is infected with some spy - warz or virii click here to get rid of it immediately before its too late http : / / jhxa . myspyerase . biz / ? id = dmv 69 remove http : / / otwm . nhjdgfm . info / bksmhr ? p 2 lor 4 jjwnwi 5 ppbgzlhrirait @ bruce - guenter . dyndns . org"], ["Subject: MY GREETINGS Hello Excuse me for this way of contacting you, I just saw your profile and I said to myself that you are the person that I am I am Mrs Marie PERRINE of French nationality, hospitalized in Europe for health reasons. I have a brain tumor and the result of some of my medical tests showed that my days on earth are numbered. Unfortunately I have no family nor child who will benefit from this money. I was advised by the Father of my church and spiritual guide to make a donation so that the LORD forgive me my sins because I had to perform illegal trafficking in various areas during my travels. You are therefore the beneficiary of 1,400,000 Euros. I offer it to you from the bottom of heart. Please accept it and put it to good use. I'm just asking for prayers so that my operation goes very well. So please write to my personal email marieperrine20@gmail.com to get in possession of your donation because it already belongs to you here. May the blessings be with you Cordially"]]

iface = gr.Interface(fn=id_email_type, inputs=gr.inputs.Textbox(lines=10, placeholder="Please paste your email subject line and body in here"), outputs=gr.outputs.Label(), title="Spam Predictor", examples = examples, description="Here is my spam predictor! Below are 4 examples to demonstrate its effectiveness. The first 2 pieces of text are not spam (ham) and the last 2 are spam. You could even try it for yourself and paste your own emails in to the textbox!", live=True).launch(share = True)