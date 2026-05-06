
# Import the needed libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# add line chart for silhoutte score
# add a link in profile 
# Thee PCA shows the clusters that we are seeing 

# -- ------------------------- Step 1: Build Streamlit App Layout ------------------------- #
# Add Main Title and Descriptions to streamlit interface
st.title("Unsupervised Machine Learning Models")
st.markdown(" **Name:** Adelyn Clemmer **| Class:** Intro to Data Science ")
st.text("Upload a CSV or Excel file to get started. Files with size over 10,000KB will cause program to run slow.")

# Allow user to uploead files
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])
tab1, tab2, tab3, tab4 = st.tabs(["🤍Your Dataframe🤍", "🤍Your Features🤍", "🤍 Hierarchical Clustering Model🤍", "🤍 🤍"]) 
    # ------------------------- Step 2: Load Uploaded Data ------------------------- #

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
# Display the dataframe and some key summary statistics about the data 
    with tab1:
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

    

    fig, axes = plt.subplots(
        nrows=(len(selected_features) + 2) // 3,
        ncols=3,
        figsize=(14, 4 * ((len(selected_features) + 2) // 3))
        )

    with tab2:
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

    # ------------------------- Step 6.5: PCA for Visualization ------------------------- #

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_


    # ------------------------- Step 7: Build the Tree ------------------------- #

    from scipy.cluster.hierarchy import linkage, dendrogram

    with tab3:
        with st.expander("🤍 What is Hierarchical Clustering 🤍"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### What is Hierarchical Clustering?")
                st.markdown("""
                Hierarchical clustering builds a **Dendrogram of Clusters** 
                by merging similar observations together.
                
                This app uses **Agglomerative/Bottom-up** clustering where:
                - Each observation begins as its own cluster.  
                - The two most similar clusters become merged at every step until everything is one cluster.
                
                In the the dendrogram visual 
                the height of each branch represents the **distance between the 
                clusters that were joined at that step.**
                """)

            with col2:

                st.info("""
                **Why Use Hierarchical Clustering?**
                
                Heirarchal clustering shows the **multi-level strucutre in data that is unlabeled.**
                That means it can group data into variables without a fixed number of clusters (k).
                This model can help us find patterns or outliers for that may require further analysis 
                when preprocessing in our data set.

                """)

                st.info("""
                **Benefits of the Model**
                
                - We do not have to specify the number of clusters beforehand
                - Still applicable for may distance metrics 
                - Paractical in real world settings including market sizing, gene studies, and topic grouping
                """)


        
        # Compute the linkage matrix
        st.subheader("Hierarchical Clustering Results")
        label_col = st.selectbox(
            "Label points in diagram by __________:",
            options=["None"] + df.columns.tolist(),
            index=0
        )
          
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

        with st.expander("🤍Dendrogram Overview🤍"):
            st.markdown("""
            A **full dendrogram is the complete merge history of your data.**
            - The bottom **leafs** are all one observation
            - Each **horizontal line** is a merge between two clusters
            - The **height of the line** is the distance between what was merged
            - **Large vertical gap** are cutting where their are the most natural clusters
            """)

            # ------------------------- Step 8: Elbow + Silhouette Analysis ------------------------- #

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        # Range of candidate cluster counts
        k_range = range(2, 11)
        sil_scores = []

        for k in k_range:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            sil_scores.append(score)

        # Plot the curve
        plt.figure(figsize=(7, 4))
        plt.plot(list(k_range), sil_scores, marker="o")
        plt.xticks(list(k_range))
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Average Silhouette Score")
        plt.title("Silhouette Analysis for Agglomerative (Ward) Clustering")
        plt.grid(True, alpha=0.3)
        plt.show()

        # Print best k
        best_k = list(k_range)[np.argmax(sil_scores)]
        #print(f"Best k by silhouette: {best_k}  (score={max(sil_scores):.3f})")
        
        
    # --- Silhouette Diagram ---
    with st.expander("🤍 Hypertune by K 🤍"):
    
    # ---- NEW: Hypertune k ---- #
        st.subheader("Hypertune Number of Clusters (k)")

        col1, col2 = st.columns([2, 1])
        with col1:
            chosen_k = st.slider(
                "Choose number of clusters (k)",
                min_value=2, max_value=20, value=best_k,
                help="Defaults to recommended k — drag to override."
            )
        with col2:
            st.metric("Recommended k", best_k)
            st.metric("Your chosen k", chosen_k)


        # PCA at chosen_k
        chosen_labels = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward").fit_predict(X_scaled)

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=chosen_labels, cmap='viridis',
                            s=60, edgecolor='k', alpha=0.7)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title(f'Agglomerative Clustering @ k={chosen_k}')
        ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.grid(True)

        if label_col != "None":
            for i, label in enumerate(df[label_col].astype(str).values):
                ax.annotate(label[:4], (X_pca[i, 0], X_pca[i, 1]),
                            fontsize=7, alpha=0.75, xytext=(4, 2),
                            textcoords="offset points")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("💡 PCA Description"):
            st.markdown("""
            This **PCA scatter plot** plots our high-dimensional data into 2D for graph for visualization.
            - Each of these **dot** represents a row in the dataset
            - Each **Color** indicates which cluster it belongs to
            - When there is **Tight, well-separated blobs** this means the data has indicate strong clusters
            - When there is **Overlapping colors** the clusters may not be well defined at this k
                        
            Try a k value that is not the recommended level, and see how they merge!
            """)


    with st.expander("🤍 Silhouette Analysis 🤍"):

        from sklearn.metrics import silhouette_score, silhouette_samples

        sample_sil = silhouette_samples(X_scaled, chosen_labels)
        avg_sil = silhouette_score(X_scaled, chosen_labels)
        max_sil = max(sil_scores)
        max_sil_k = list(k_range)[np.argmax(sil_scores)]

        # --- Metric Cards ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chosen k", chosen_k)
        col2.metric("Chosen k silhouette score", f"{avg_sil:.3f}")
        col3.metric("Best k", max_sil_k)
        col4.metric("Best silhouette score", f"{max_sil:.3f}")

        # --- Silhouette Diagram ---
        plt.figure(figsize=(7, 4))
        plt.plot(list(k_range), sil_scores, marker="o")
        plt.xticks(list(k_range))
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Average Silhouette Score")
        plt.title("Silhouette Analysis for Agglomerative Clustering")
        plt.axvline(chosen_k, color="red", linestyle="--", alpha=0.6, label=f"chosen k={chosen_k}")
        plt.axvline(max_sil_k, color="green", linestyle="--", alpha=0.6, label=f"best k={max_sil_k}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

        with st.expander("💡 Silhouette Overview "):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**What this chart shows**")
                st.markdown("""
                This plot shows the average silhouette score for each possible 
                number of clusters (k) from 2 to 10.

                - The **peak of the curve** is where clusters are most 
                distanced from each other which becomes the model's recommended k
                - A **higher score** means points are firmly tight when in their own 
                cluster and seperated from neighboring clusters
                - A **flatter curve** means k choice matters less for your data
                """)

            with col2:
                st.markdown("**Chosen k vs best k**")
                st.markdown(f"""
                The model for this data set recommends **k={max_sil_k}** which is displayed by the green line based on 
                the highest silhouette score of **{max_sil:.3f}**.

                You have chosen **k={chosen_k}** (red line), which scores 
                **{avg_sil:.3f}**.
                """)


