import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Illustrating the Central Limit Theorem with Streamlit')
st.subheader('An App by Balablu Bulaba Bubu baby')
st.write(('This app simulates a thousand coin flips using the chance of heads input below,'
'and then samples with replacement from that population and plots the histogram of the'
'means of the samples, in order to illustrate the Central Limit Theorem!'))

perc_heads= st.number_input(label='Chance of Coins Landing on Heads', min_value=0.0, max_value=1.0, value=.5)
graph_title= st.text_input(label='Graph Title')

binom_dist = np.random.binomial(1, perc_heads, 1000)

st.write('Hello World, Esther is Medusa, very evil!!!')
st.write('vetg')
st.write('vetg')
st.write(np.mean(binom_dist))

list_of_means = []
for i in range(0, 1000):
    list_of_means.append(np.random.choice(binom_dist, 100, replace = True).mean())

fig1, ax1 = plt.subplots()
ax1 = plt.hist(list_of_means, range=[0, 1])
plt.title(graph_title)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2 = plt.hist([1,1,1,1]) 
st.pyplot(fig2)

st.button("Re-run")