# IMPORTS
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

## HELPER Functions 
# Load data from external source
@st.cache
def load_df():
    df = pd.read_csv("https://raw.githubusercontent.com/ThuwarakeshM/PracticalML-KMeans-Election/master/voters_demo_sample.csv")
    return df

df = load_df()

# KMeans function
def run_kmeans(df, n_clusters = 2):
    kmeans = KMeans(n_clusters, random_state = 24).fit(df[["Age", "Income"]])
    fig, ax = plt.subplots(figsize = (16,9))
    ax.xaxis.label.set(fontsize = 20)
    ax.yaxis.label.set(fontsize = 20)
    
    # Scatter plot
    ax = sns.scatterplot(
        ax = ax,
        x = df.Age,
        y = df.Income,
        hue = kmeans.labels_,
        palette = sns.color_palette("husl", n_colors = n_clusters),
        legend = None)
    
    # Annotate the clusters
    for idx, (age, income) in enumerate(kmeans.cluster_centers_):
        ax.scatter(age, income)
        ax.annotate(
            f"Cluster {idx+1}",
            xy = (age,income),
            fontsize = 20,
            xytext = (age + 3, income+3),
            bbox = dict(boxstyle = "square,pad=0.3", fc = "white"),
            ha = "center",
            va = "center"
        )
    return fig

## -- SIDEBAR --

sidebar = st.sidebar
df_display = sidebar.checkbox("Display Row Data", value = True)
n_clusters = sidebar.slider("Select Number of Clusters",
                            min_value = 2,
                            max_value = 10)
                          
sidebar.write(
    """
    Hey there!  
    I am Mani Chandana, a data geek.  
    
    This is my first Streamlit deployed ML model (KMeans)!   
    I'm so proud of finally deploying my model on a web app.  
    
    Lookout for more!
    """
)

## --- MAIN ---
# Title
st.title("Interactive k-means clustering")

# Scatter plot of cluster
st.write(run_kmeans(df,n_clusters))

# A description
st.write("Here is the dataset used in this analysis:")

# Display the dataframe
if df_display:
     st.write(df)
        
        
                         