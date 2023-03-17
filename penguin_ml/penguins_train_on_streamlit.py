# import necessary modules
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# title and instructions
st.title('Penguin Classifier: A Machine Learning App')
st.write("This app uses 6 inputs to predict the species of penguin using "
         "a model built on the Palmer's Penguin's dataset. "
         "Use the form below to get started!")

# password the file
password_guess = st.text_input('What is the Password?')

if password_guess != '' or st.secrets["password"]:
    st.write('Your password is not Nkereke so no Townhall for you.')
    st.stop()

elif password_guess == '':
    st.stop()

# instantiate general used variable as None
penguin_df= None
unique_penguin_mapping = None

# upload file
penguin_file = st.file_uploader('Upload your own penguin data')

# handle no file upload
if penguin_file is None:
    # use the already build model
    rf_pickle = open('penguin_ml/random_forest_penguin.pickle', 'rb')
    map_pickle = open('penguin_ml/output_penguin.pickle', 'rb')

    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)

    rf_pickle.close()
    map_pickle.close()
else:   
    # read data, clean it and train model

    # read data 
    penguin_df = pd.read_csv(penguin_file)
    penguin_df.dropna(inplace=True)
    output = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
    'flipper_length_mm', 'body_mass_g', 'sex']]

    # one-hot encode categorical features
    features= pd.get_dummies(features)
    output, uniques = pd.factorize(output)

    # split data into training and test set
    x_train, x_test, y_train, y_test = train_test_split(features, 
    output, test_size=0.8)

    # train random forest classifier model
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)
    y_pred= rfc.predict(x_test)
    score= accuracy_score(y_pred, y_test)

    st.write('We trained a Random Forest model on these data, '
             'it has a score of {} ! Use the inputs below to'
              ' try out the model.'.format(score))
    
    fig, ax = plt.subplots()
    ax = sns.barplot(x= rfc.feature_importances_, y= features.columns)
    plt.title('Which features are the most important for species prediction?')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    fig.savefig('feature_importance.png')
    
# Receiving user input
with st.form('user_inputs'):
    island = st.selectbox('Penguin Island', options= ['Biscoe', 'Dream', 
                                                    'Torgerson'])
    sex = st.selectbox('Sex', options= ['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value= 0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value= 0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value= 0)
    body_mass = st.number_input('Body Mass (g)', min_value= 0)
    st.form_submit_button()

# setting data into the correct format
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

# making sure nothing happens until user input is given
if [island, sex, bill_length, bill_depth, flipper_length, body_mass] == ['Biscoe', 'Female', 0, 0, 0, 0]:
    st.write('Please tweak the inputs.')
    st.stop()

else:
    # making predictions
    new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,
                                body_mass, island_biscoe, island_dream,
                                island_torgerson, sex_female, sex_male]])

    #mapping prediction to its species
    if unique_penguin_mapping is None:
        unique_penguin_mapping = uniques

    prediction_species = unique_penguin_mapping[new_prediction][0]

    # display prediction
    st.subheader("Predicting Your Penguin's Species:")

    st.write('- Your inputs are {} '.format([island, sex, bill_length, 
                                            bill_depth, flipper_length, 
                                            body_mass]))

    st.write('- We predict your penguin is of the {} species'.format(
        prediction_species))

    st.write('- We used a machine learning (Random Forest) model to '
            'predict the species, the features used in this prediction'
            ' are ranked by relative importance below.')

    # model understanding with visualizations

    # random forest feature importance graph
    st.image('penguin_ml/feature_importance.png')

    # Histogram plots of each feature and line plot of user input
    st.write('Below are the histograms for each continuous variable'
            ' seperated by penguin species. The vertical line represents '
            'your inputted value.')

    if penguin_df is None:
        penguin_df = pd.read_csv('penguin_ml/penguins.csv')
        penguin_df.dropna(inplace=True)

    def plot_features(feature_name, user_input_feature):
        fig, ax = plt.subplots()
        ax = sns.displot(x=penguin_df[feature_name], hue= penguin_df['species'])
        plt.axvline(user_input_feature)
        plt.title((feature_name.title() + ' by Species'))
        return st.pyplot(ax)

    plot_features('bill_length_mm', bill_length)
    plot_features('bill_depth_mm', bill_depth)
    plot_features('flipper_length_mm', flipper_length)

    # fig, ax = plt.subplots()
    #     ax = sns.displot(x=penguin_df['bill_length_mm'], hue= penguin_df['species'])
    #     plt.axvline(bill_length)
    #     plt.title('Bill Length by Species')
    #     st.pyplot(ax)

    # fig, ax = plt.subplots()
    # ax = sns.displot(x=penguin_df['bill_depth_mm'], hue= penguin_df['species'])
    # plt.axvline(bill_depth)
    # plt.title('Bill Depth by Species')
    # st.pyplot(ax)

    # fig, ax = plt.subplots()
    # ax = sns.displot(x=penguin_df['flipper_length_mm'], hue= penguin_df['species'])
    # plt.axvline(flipper_length)
    # plt.title('Flipper Length by Species')
    # st.pyplot(ax)