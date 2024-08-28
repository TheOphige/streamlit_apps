import streamlit as st
import pandas as pd

# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# import seaborn as sns
# import datetime as dt
# from bokeh.plotting import figure
# import altair as alt
import pydeck as pdk

st.title("SF Trees")
st.write(
    "This app analyses trees in San Francisco using"
    " a dataset kindly provided by SF DPW"
)
trees_df = pd.read_csv("trees.csv")
# df_dbh_grouped= pd.DataFrame(trees_df.groupby(['dbh']).count()['tree_id'])
# df_dbh_grouped.columns = ['tree_count']
# st.write(df_dbh_grouped.head())

# st.line_chart(df_dbh_grouped)
# st.bar_chart(df_dbh_grouped)
# st.area_chart(df_dbh_grouped)

# df_dbh_grouped['new_col'] = np.random.randn(len(df_dbh_grouped)) * 500
# st.write(df_dbh_grouped.head())

# st.line_chart(df_dbh_grouped)

# st.subheader('streamlit map')
# trees_df = trees_df.dropna(subset=['longitude', 'latitude'])
# trees_df = trees_df.sample(n = 1000)
# st.map(trees_df)

# st.subheader('Plotly chart')
# fig = px.histogram(trees_df['dbh'])
# st.plotly_chart(fig)

# trees_df['age'] = (pd.to_datetime('today') - pd.to_datetime(trees_df['date'])).dt.days

# st.subheader('Seaborn Chart')
# fig_sb, ax_sb = plt.subplots()
# ax_sb = sns.histplot(trees_df['age'])
# plt.xlabel('Age (Days)')
# st.pyplot(fig_sb)

# st.subheader('Matplotlib Chart')
# fig_mpl, ax_mpl = plt.subplots()
# ax_mpl = plt.hist(trees_df['age'])
# plt.xlabel('Age (Days)')
# st.pyplot(fig_mpl)

# st.subheader('Bokeh Chart')
# scatterplot = figure(title= 'Bokeh Scatterplot')
# scatterplot.scatter(trees_df['dbh'], trees_df['site_order'])
# scatterplot.xaxis.axis_label = "dbh"
# scatterplot.yaxis.axis_label = "site_order"
# st.bokeh_chart(scatterplot)

# df_caretaker = trees_df.groupby(['caretaker']).count()['tree_id'].reset_index()
# df_caretaker.columns = ['caretaker', 'tree_count']
# fig = alt.Chart(df_caretaker).mark_bar().encode(x= 'caretaker', y= 'tree_count')
# st.altair_chart(fig)
# OR
# fig = alt.Chart(trees_df).mark_bar().encode(x= 'caretaker', y= 'count(*):Q')
# st.altair_chart(fig)

trees_df.dropna(how="any", inplace=True)
sf_initial_view = pdk.ViewState(latitude=37.77, longitude=-122.4, zoom=11, pitch=30)
# sp_layer = pdk.Layer(
#     'Scatterplot',
#     data=trees_df,
#     get_position= ['longitude', 'latitude'],
#     get_radius=30
# )
hx_layer = pdk.Layer(
    "HexagonLayer",
    data=trees_df,
    get_position=["longitude", "latitude"],
    get_radius=100,
    extruded=True,
)
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=sf_initial_view,
        layers=[hx_layer],
    )
)
