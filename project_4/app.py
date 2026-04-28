
# Import the needed libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# -- ------------------------- Step 1: Build Streamlit App Layout ------------------------- #
# Add Main Title and Descriptions to streamlit interface
st.title("Unsupervised Machine Learning Models")
st.markdown(" **Name:** Adelyn Clemmer **| Class:** Intro to Data Science ")
st.text("Upload a CSV or Excel file to get started. Files with size over 10,000KB will cause program to run slow.")

# Allow user to uploead files
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

# ------------------------- Step 2: Load Uploaded Data ------------------------- #

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
       

    st.subheader("Preview of Uploaded Data")
 # Display the dataframe and some key summary statistics about the data 
    with st.expander("🤍Your Dataframe🤍"):
        st.dataframe(df)
        # Create a side by side layout
        col1, col2, col3 = st.columns(3)

        with col1:
            # display the number of missing data points in a set
            st.markdown("Missing Data")
            missing = df.copy().isnull().sum().reset_index()
            missing.columns = ["Column", "Number of Missing Values"]
            # sort the missing values in descending order and display
            missing = missing.sort_values(by="Number of Missing Values", ascending=False)
            st.dataframe(missing, use_container_width=True)

        with col2:
            #  use a describe() to show the summary statistics of the dataframe
            st.markdown("Data Preview")
            st.dataframe(df.describe(), use_container_width=True)

        with col3:
            # describe all the data types in each column to help the user understand the structure of their data 
            # and make informed decisions about feature selection and model building
            st.markdown("Data Types")
            dtypes = df.dtypes.reset_index()
            dtypes = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.values})
            st.dataframe(dtypes, use_container_width=True)



 # ------------------------- Step 3: Let User Drop Columns ------------------------- #

    st.sidebar.subheader("Data Cleaning")

    # ------------------------- Step 4: Convert Boolean Columns to Numeric ------------------------- #

    object_cols = df.select_dtypes(include="object").columns.tolist()

    possible_bool_cols = []

    for col in object_cols:
        unique_vals = df[col].dropna().astype(str).str.lower().unique()
        if set(unique_vals).issubset({"true", "false"}):
            possible_bool_cols.append(col)

    bool_cols_to_convert = st.sidebar.multiselect(
        "Convert True/False columns to numeric",
        possible_bool_cols,
        default=possible_bool_cols
    )

    for col in bool_cols_to_convert:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0})
        )

  

    # ------------------------- Step 5: Select Features for Clustering ------------------------- #

    st.sidebar.subheader("Feature Selection")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    selected_features = st.sidebar.multiselect(
        "Select numeric features for clustering",
        numeric_cols,
        default=numeric_cols[:5]
    )

    if len(selected_features) < 2:
        st.warning("Please select at least two numeric features for clustering.")

    else:
        feature_df = df[selected_features].copy()

        # Missing value choice
        missing_option = st.sidebar.radio(
            "Handle missing values",
            ["Drop rows with missing values", "Fill missing values with column mean"]
        )

        if missing_option == "Drop rows with missing values":
            mask = feature_df.notna().all(axis=1)
            feature_df = feature_df[mask]
            df = df.loc[mask].copy()
        else:
            feature_df = feature_df.fillna(feature_df.mean())
            df = df.copy()

    
    st.subheader("Feature Distributions")

    fig, axes = plt.subplots(
        nrows=(len(selected_features) + 2) // 3,
        ncols=3,
        figsize=(14, 4 * ((len(selected_features) + 2) // 3))
        )

    with st.expander("🤍Your Features🤍"):
        m1, m2, m3 = st.columns(3)
        
        m1.metric("Numeric feature columns", len(selected_features))
        m2.metric("Boolean Converts", len(bool_cols_to_convert))
        m3.metric("Rows with complete data", len(feature_df))
        
        feature_df.hist(ax=axes.flatten()[:len(selected_features)], edgecolor="k", bins=15)

            # Hide any unused subplot axes
        for i in range(len(selected_features), len(axes.flatten())):
            axes.flatten()[i].set_visible(False)

        plt.suptitle("Distribution of Each Selected Feature", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------- Step 6: Scale Data ------------------------- #
    
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)


    # ------------------------- Step 7: Build the Tree ------------------------- #

    from scipy.cluster.hierarchy import linkage, dendrogram

    # Compute the linkage matrix
    Z = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(20, 7))
    dendrogram(Z, labels=None, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Row Label")
    ax.set_ylabel("Distance")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
