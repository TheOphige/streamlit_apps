import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.title("Palmer's Penguins")
st.markdown("Use this Streamlit app to make your own scatterplot about penguins!")


penguin_file = st.file_uploader("Select Your Local Penguins CSV(default provided)")


@st.cache
def load_file(penguin_file):
    time.sleep(3)
    if penguin_file is not None:
        penguins_df = pd.read_csv(penguin_file)
    else:
        penguins_df = pd.read_csv("penguins.csv")
        # st.stop()
    return penguins_df


penguins_df = load_file(penguin_file)

selected_x_var = st.selectbox(
    "What do want the x variable to be?",
    ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
)
selected_y_var = st.selectbox(
    "What about the y?",
    ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
)
selected_gender = st.selectbox(
    "What gender do you want to filter for?",
    ["allpenguins", "male penguins", "female penguins"],
)

if selected_gender == "male penguins":
    penguins_df = penguins_df[penguins_df["sex"] == "male"]
elif selected_gender == "female penguins":
    penguins_df = penguins_df[penguins_df["sex"] == "female"]
else:
    pass


sns.set_style("darkgrid")
markers = {"Adelie": "X", "Gentoo": "s", "Chinstrap": "o"}

fig, ax = plt.subplots()
ax = sns.scatterplot(
    data=penguins_df,
    x=selected_x_var,
    y=selected_y_var,
    hue="species",
    markers=markers,
    style="species",
)
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title("Scatterplot of Palmer's Penguins: {}".format(selected_gender))

st.pyplot(fig)
