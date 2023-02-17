import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
binom_dist = np.random.binomial(1, .5, 1000)

st.write('Hello World, Esther is Medusa, very evil!!!')
st.write('vetg')
st.write('vetg')
st.write(np.mean(binom_dist))

list_of_means = []
for i in range(0, 1000):
    list_of_means.append(np.random.choice(binom_dist, 100, replace = True).mean())

#fig, ax = plt.subplots()
#ax = plt.hist(list_of_means)
# st.pyplot(fig)

plt.hist(list_of_means)
st.pyplot()
plt.hist([1,1,1,1]) 
st.pyplot()

st.button("Re-run")