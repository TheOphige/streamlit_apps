import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Palmer's Penguins")

st.markdown('Use this Streamlit app to make your own scatterplot about penguins!')

#selected_species= st.selectbox('What species would you like to visualize?',
#['Adelie', 'Gentoo', 'Chinstrap'])
selected_x_var= st.selectbox('What do want the x variable to be?',
['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm','body_mass_g'])
selected_y_var = st.selectbox('What about the y?',
['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm','body_mass_g'])

#import our data
penguins_df = pd.read_csv('penguins.csv')
#st.write(penguins_df.head())

#penguins_df= penguins_df[penguins_df['species'] == selected_species]

sns.set_style('darkgrid')
markers= {"Adelie": "X", "Gentoo": "s", "Chinstrap":'o'}

fig, ax = plt.subplots()
ax = sns.scatterplot(data=penguins_df, x= selected_x_var, 
y= selected_y_var, hue='species', markers= markers, style='species')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
#plt.title('Scatterplot of {} Penguins'.format(selected_species))
plt.title("Scatterplot of Palmer's Penguins")

st.pyplot(fig)