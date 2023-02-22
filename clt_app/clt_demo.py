import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

perc_heads= st.number_input(label='Chance of Coins Landing on Heads', min_value=0.0, max_value=1.0, value=.5)

binom_dist = np.random.binomial(1, perc_heads, 1000)

st.write('Hello World, Esther is Medusa, very evil!!!')
st.write('vetg')
st.write('vetg')
st.write(np.mean(binom_dist))

list_of_means = []
for i in range(0, 1000):
    list_of_means.append(np.random.choice(binom_dist, 100, replace = True).mean())

fig1, ax1 = plt.subplots()
ax1 = plt.hist(list_of_means)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2 = plt.hist([1,1,1,1]) 
st.pyplot(fig2)

st.button("Re-run")