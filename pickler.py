import pandas as pd
import pickle as pick
import matplotlib.pyplot as plt

training_data_csv = 'C:\\Users\\hayde\\OneDrive\\Documents\\Final_Project_497\\final_code\\data\\training_data.csv'

df = pd.read_csv(training_data_csv, sep=',', usecols=['email', 'phishing', 'label'])
df = pd.DataFrame(df)



#so I can see the balances of the dataset
def visualize_data(labels):
    labels = df['label']
    l_counts = labels.value_counts()
    plt.pie(l_counts, labels=l_counts.index, autopct='%.2f')
    plt.show()

#pickle the csv
def pickling_training_data():
    data = df['email']
    labels = df['label']
    labelsfile = 'C:\\Users\\hayde\\OneDrive\\Documents\\Final_Project_497\\final_code\\data\\training_labels.pkl'
    datafile = 'C:\\Users\\hayde\\OneDrive\\Documents\\Final_Project_497\\final_code\\data\\training_data.pkl'

    labelsout = open(labelsfile, 'wb')
    pick.dump(labels, labelsout)
    labelsout.close()

    dataout = open(datafile, 'wb')
    pick.dump(data, dataout)
    dataout.close()

    # visualize_data(labels)

    return labelsfile, datafile