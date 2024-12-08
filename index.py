import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Titanic Themed Background Styling
def set_titanic_theme():
    st.markdown(
        """
        <style>
        /* Background Gradient */
        .stApp {
            background: linear-gradient(120deg, #002f4b, #dcdddf);
            color: white;
        }

        /* Sidebar Background */
        section[data-testid="stSidebar"] {
            background-color: #004e7c;
            border-right: 2px solid white;
        }

        /* Text Colors */
        .stTextInput > div > label, .stButton > button {
            color: #fff !important;
        }

        /* Input and Button Styling */
        .stTextInput > div > input {
            background-color: #007ea7;
            color: white !important;
            border: 2px solid #002f4b !important;
            border-radius: 5px !important;
        }
        .stButton > button {
            background-color: #005082;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        .stButton > button:hover {
            background-color: #007ea7;
        }

        /* Table Styling */
        .dataframe {
            background-color: white;
            color: black;
            border-radius: 10px;
        }

        /* Plot Background */
        div[data-testid="stPlotlyChart"], div[data-testid="stImage"] > img {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Apply the Titanic Theme
set_titanic_theme()

# Page Title
st.title("🚢 Titanic Dashboard")

# Sidebar for Login
st.sidebar.header("Login")

# Create a session state variable for the username
if "username" not in st.session_state:
    st.session_state.username = None

# Login Section
if not st.session_state.username:
    username = st.sidebar.text_input("Enter a username")
    if st.sidebar.button("Log In"):
        if username:
            st.session_state.username = username
            st.sidebar.success(f"Logged in as: {st.session_state.username}")
        else:
            st.sidebar.error("Please enter a username.")
else:
    st.sidebar.write(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Log Out"):
        st.session_state.username = None

# Main Application Logic
if st.session_state.username:
    # File Upload Section
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Browse files", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded dataset
        data = pd.read_csv(uploaded_file)
        st.write(f"### Welcome, {st.session_state.username}!")
        st.write("### Preview of the Uploaded Dataset:")
        st.dataframe(data.head())

        # Graph Stats: Sex and Age
        if "Sex" in data.columns:
            st.write("### Sex Distribution")
            sex_counts = data["Sex"].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=sex_counts.index, y=sex_counts.values, ax=ax, palette="viridis")
            ax.set_title("Sex Distribution")
            ax.set_xlabel("Sex")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning("The dataset does not contain a 'Sex' column.")

        if "Age" in data.columns:
            st.write("### Age Group Distribution")
            # Create age groups
            data["Age Group"] = pd.cut(data["Age"], bins=[0, 18, 30, 45, 60, 100], 
                                       labels=["0-18", "19-30", "31-45", "46-60", "60+"])
            age_group_counts = data["Age Group"].value_counts().sort_index()
            fig, ax = plt.subplots()
            sns.barplot(x=age_group_counts.index, y=age_group_counts.values, ax=ax, palette="viridis")
            ax.set_title("Age Group Distribution")
            ax.set_xlabel("Age Group")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning("The dataset does not contain an 'Age' column.")

        # Data Preparation for Association Rule Mining
        st.write("### Preparing Data for Association Rule Mining")
        st.write("Dataset dimensions:", data.shape)

        # Ensure binary encoding for the dataset
        def preprocess_data(df):
            """
            Preprocess the dataset to create a one-hot encoded binary matrix.
            Assumes the dataset contains transactional data where each column
            is an item, and the value indicates the quantity or presence.
            """
            # Attempt to convert all values to numeric; non-convertible values will be set to NaN
            df = df.apply(pd.to_numeric, errors="coerce")

            # Fill NaN values with 0
            df.fillna(0, inplace=True)

            # Convert all positive values to 1 (presence) and zeros/negatives to 0 (absence)
            binary_encoded_df = df.applymap(lambda x: 1 if x > 0 else 0)

            return binary_encoded_df

        # Handle missing values and preprocess the data
        preprocessed_data = preprocess_data(data)
        st.write("Preprocessed Data (Binary Encoded):")
        st.dataframe(preprocessed_data.head())

        # Perform Frequent Itemsets Mining
        st.write("### Frequent Itemsets Mining")
        min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.05)
        frequent_itemsets = apriori(preprocessed_data, min_support=min_support, use_colnames=True)
        st.write(frequent_itemsets)

        # Generate Association Rules
        st.write("### Association Rules")
        metric = st.sidebar.selectbox("Select Metric for Rule Filtering", ["support", "confidence", "lift"])
        threshold = st.sidebar.slider("Threshold", 0.1, 1.0, 0.5)

        # Calculate the number of itemsets
        num_itemsets = len(frequent_itemsets)

        # Pass num_itemsets to association_rules()
        rules = association_rules(frequent_itemsets, num_itemsets=num_itemsets, metric=metric, min_threshold=threshold)
        st.write(rules)

        # Visualize Rules
        st.write("### Visualizations")
        st.write("#### Network Diagram of Rules")
        G = nx.DiGraph()
        for index, rule in rules.iterrows():
            antecedent = tuple(rule["antecedents"])
            consequent = tuple(rule["consequents"])
            G.add_edge(antecedent, consequent, weight=rule[metric])

        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
        st.pyplot(plt)

        st.write("#### Support and Confidence Bar Chart")
        plt.figure(figsize=(10, 6))
        rules.plot(kind="bar", x="support", y="confidence", title="Support vs Confidence")
        st.pyplot(plt)

        # Recommendations
        st.write("### Product Recommendations")
        st.write("Explore frequently bought-together products and optimize cross-selling strategies.")

    else:
        st.write("Please upload a dataset to begin!")
else:
    st.write("Please log in to access the application.")
