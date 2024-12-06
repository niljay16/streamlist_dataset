import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

def mine_association_rules(data, transaction_col, item_col, min_support, metric, min_threshold):
    
    if transaction_col not in data.columns or item_col not in data.columns:
        raise ValueError(f"Dataset must contain '{transaction_col}' and '{item_col}' columns.")
    
    basket = data.groupby([transaction_col, item_col])[item_col].count().unstack().fillna(0)
    basket = (basket > 0).astype(int)

  
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return frequent_itemsets, rules


def plot_frequent_itemsets(frequent_itemsets, top_n=10):
    top_items = frequent_itemsets.nlargest(top_n, 'support')
    fig = px.bar(top_items, x='itemsets', y='support', title='Top Frequent Itemsets', labels={'itemsets': 'Itemsets', 'support': 'Support'})
    return fig


def plot_rules_network(rules):
    G = nx.DiGraph()

    for _, rule in rules.iterrows():
        G.add_edge(tuple(rule['antecedents']), tuple(rule['consequents']), weight=rule['lift'])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(G, pos, with_labels=True, node_size=7000, font_size=10, node_color='lightblue', edge_color='gray')
    plt.title("Association Rules Network")
    return plt


def generate_recommendations(rules):
    recommendations = rules[['antecedents', 'consequents', 'support', 'confidence']].head(5)
    return recommendations




st.sidebar.title("User Login")
username = st.sidebar.text_input("Enter your username:", value="", placeholder="Type your name")


if username:
    st.sidebar.markdown(f"**Logged in as:** {username}")
else:
    st.sidebar.info("Please enter your username to proceed.")


st.title("Insights Dashboard Using Association Rule Mining")
if username:
    st.markdown(f"Welcome, **{username}**! Explore your dataset below.")
else:
    st.info("Please enter your username in the sidebar.")


if username:
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
else:
    st.info("Please provide a username to enable file upload.")

if username and uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:")
    st.dataframe(data.head())

    st.sidebar.header("Dataset Configuration")
    transaction_col = st.sidebar.text_input("Enter Transaction Column Name:", "Transaction_ID")
    item_col = st.sidebar.text_input("Enter Item Column Name:", "Item")

    st.sidebar.header("Settings")
    min_support = st.sidebar.slider("Min Support", 0.01, 1.0, 0.05)
    min_threshold = st.sidebar.slider("Min Confidence", 0.01, 1.0, 0.5)
    metric = st.sidebar.selectbox("Metric", ['confidence', 'lift', 'leverage', 'conviction'])

    try:
        frequent_itemsets, rules = mine_association_rules(data, transaction_col, item_col, min_support, metric, min_threshold)

      
        st.subheader("Frequent Itemsets")
        st.write(frequent_itemsets)

        st.subheader("Association Rules")
        st.write(rules)

       
        st.subheader("Top Frequent Itemsets")
        fig = plot_frequent_itemsets(frequent_itemsets)
        st.plotly_chart(fig)

        st.subheader("Association Rules Network")
        plt = plot_rules_network(rules)
        st.pyplot(plt)

        st.subheader("Top Product Bundles to Recommend")
        recommendations = generate_recommendations(rules)
        st.write(recommendations)

    except ValueError as e:
        st.error(f"Error: {e}")

else:
    if not username:
        st.warning("Please log in to enable dataset upload.")
    elif not uploaded_file:
        st.info("Please upload a dataset to proceed.")
