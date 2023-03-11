# importing important modules
import streamlit as st
import pickle

# title and instructions
st.title('Penguin Classifier')
st.write("This app uses 6 inputs to predict the species of penguin using"
         "a model built on the Palmer's Penguin's dataset."
         "Use the form below to get started!")

# loading the model and outputs file
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')

rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)

rf_pickle.close()
map_pickle.close()

# Receiving user input
island = st.selectbox('Penguin Island', options= ['Biscoe', 'Dream', 
                                                  'Torgerson'])
sex = st.selectbox('Sex', options= ['Female', 'Male'])
bill_length = st.number_input('Bill Length (mm)', min_value= 0)
bill_depth = st.number_input('Bill Depth (mm)', min_value= 0)
flipper_length = st.number_input('Flipper Length (mm)', min_value= 0)
body_mass = st.number_input('Body Mass (g)', min_value= 0)

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

# making predictions
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,
                               body_mass, island_biscoe, island_dream,
                               island_torgerson, sex_female, sex_male]])

#mapping prediction to its species
prediction_species = unique_penguin_mapping[new_prediction][0]

# display prediction
st.subheader("Predicting Your Penguin's Species:")
st.write('The user inputs are {}'.format([island, sex, bill_length, 
                                          bill_depth, flipper_length, 
                                          body_mass]))

st.write('We predict your penguin is of the {} species'.format(
    prediction_species))

st.write('We used a machine learning (Random Forest) model to '
         'predict the species, the features used in this prediction'
         'are rankedby relative importance below.')

# model understanding with visualizations

# random forest feature importance graph
st.image('feature_importance.png')

# Histogram plots of each feature and line plot of user input
st.write('Below are the histograms for each continuous variable'
         ' seperated by penguin species. The vertical line represents '
         'your inputted value.')

penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'], hue= penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_depth_mm'], hue= penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'], hue= penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Flipper Length by Species')
st.pyplot(ax)