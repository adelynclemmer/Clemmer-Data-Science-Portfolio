
# ------------------------- Step 1: Import Libraries ------------------------- #
# Import all necessary libraries for data manipulation, visualizations, and unsupervised modeling.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram


# ---------------------------- Step 2: Build Streamlit App Layout with File Upload ------------------------- #
# Build the main interface with titles, descriptions, sidebar uploaders, and tabs.

# Add Main Title and Descriptions to streamlit interface
st.title("Unsupervised Machine Learning Models")
st.markdown(" **Name:** Adelyn Clemmer **| Class:** Intro to Data Science ")
st.text("Upload a CSV or Excel file to get started. Files with size over 10,000KB will cause program to run slow.")

# Allow user to uploead files
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

# Organizational Structure to include raw data preview, specific feature analysis and our two models
tab1, tab2, tab3, tab4 = st.tabs(["🤍Your Dataframe🤍", "🤍Your Features🤍", "🤍 Hierarchical Clustering Model🤍", "🤍 K- Means🤍"]) 
    
    
# ---------------------------- Step 3: Load Uploaded Data ------------------------- #
if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    

# ------------------------- Step 4: Data Cleaning and Boolean Conversion ------------------------- #
    ## Allow users to eliminate features in their data set for modeling and covert the boolean values to numeric for the model
    st.sidebar.subheader("Data Cleaning")

    ## Identify object columns and check for True/False values to convert to numeric for modeling
    object_cols = df.select_dtypes(include="object").columns.tolist()
    possible_bool_cols = []

    ## Loop through object columns to find those that can be converted to boolean
    for col in object_cols:
        unique_vals = df[col].dropna().astype(str).str.lower().unique()
        if set(unique_vals).issubset({"true", "false"}):
            possible_bool_cols.append(col)

    ## Allow users to select which boolean columns to convert to numeric for modeling
    bool_cols_to_convert = st.sidebar.multiselect(
        "Convert True/False columns to numeric",
        possible_bool_cols,
        default=possible_bool_cols
    )

    ## Convert selected boolean columns to numeric for unsupervised learning models because both rely on 
    ## distance calculations and need numeric data to compute distances between data points
    for col in bool_cols_to_convert:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0})
        )

# ------------------------- Step 5: Select Features for Clustering ------------------------- #

    # Allow users to select which numeric features to include in the clustering models.
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    selected_features = st.sidebar.multiselect(
        "Select numeric features for clustering",
        numeric_cols,
        default=numeric_cols[:5]
    )
    # if the user selects less than 2 features, show a warning message because clustering models need at 
    # least 2 features to compute distances and form meaningful clusters.
    if len(selected_features) < 2:
        st.warning("Please select at least two numeric features for clustering.")
    else:
        feature_df = df[selected_features].copy()

    # Allow the user to decide how to handle missing values in the selected features 
    # both K-Means and Hierarchical clustering models require complete data to compute distances accurately but 
    # giving users the option to either drop rows with missing values or fill them with the column mean helps preserve 
    # more discrecion. If users have a set that supports filling by mean they can keep more data points for modeling,
    # but if they have a set with many missing values they may want to drop those rows to avoid skewing the results.
    missing_option = st.sidebar.radio(
        "Handle missing values",
        ["Drop rows with missing values", "Fill missing values with column mean"]
    )

    if missing_option == "Drop rows with missing values":
        # Create a mask to identify rows with complete data across all selected features
        mask = feature_df.notna().all(axis=1)
        feature_df = feature_df[mask]
        # Update the main dataframe to only include rows with complete data for the selected features
        df = df.loc[mask].copy()
    else:
        feature_df = feature_df.fillna(feature_df.mean())
        df = df.copy()

# ---------------------------- Step 6: Display Dataframe and User Information ------------------------- #
    with tab1:
        st.dataframe(df)
        # Create a side by side layout
        col1, col2, col3 = st.columns(3)

        with col1:
            # display the number of missing data points in an uploaded set
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


# ---------------------------- Step 7: Visualize Feature Distributions ------------------------- #
## We include feature distribution visualizations to help users understand the scale and shape of their data before modeling. 
# Both K-Means and Hierarchical clustering rely on distance calculations, so features with different scales or skewed distributions can impact how clusters are formed. 
# By visualizing the distributions, users can identify why features need to be scaled or if there are outliers 
    
    with tab2:
        m1, m2, m3 = st.columns(3)
        
        m1.metric("Numeric feature columns", len(selected_features))
        # we show the number of boolean columns that were converted to numeric to help users understand how their features have changed
        m2.metric("Boolean Converts", len(bool_cols_to_convert))

        m3.metric("Rows with complete data", len(feature_df))

        with st.expander("💡 Why Do We Look at Feature Distributions? 💡"):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### What These Graphs Show")
                st.markdown("""
                Each histogram shows how the values of a single feature are 
                **distributed across your dataset**

                What we may observe:
                - **Bell-shaped curve:** Values are normally distributed and 
                centered around a mean a model like K-Means does well with these 
                - **Skewed distribution:** Values are bunched to one side with 
                a long tail which can create bias clustering toward the extreme values
                - **Multiple peaks:** The feature may already have natural 
                subgroups hiding in it, a main reason we use unsupervised clustering models in the first place
                - **Very flat distribution:** If values are spread evenly with no 
                clear center the feature may not add much signal to the model
                """)

            with col2:
                st.markdown("### Why It Matters for Clustering")
                st.markdown("""
                K-Means and Hierarchical clustering rely on **distance calculations** 
                to decide which points belong together. That means the scale and shape 
                of each feature directly affects how the model groups your data.
                """)

                st.info("""
                **Why Scale Before Clustering?**
                
                If a feature rages from 0–10,000 it will dominate distance calculations 
                over a feature ranging from 0–1, even if the smaller feature is 
                more meaningful. Viewing distributions first helps you spot these 
                imbalances before the model runs.
                """)

                st.info("""
                **Why Outliers Matter**
                
                A single extreme value can stretch a feature's range and result in pulling 
                centroids away from the true cluster center. Check for long tails or isolated bars 
                in your data. That feature may contain outliers that could distort your clustering results.
                """)

        # Plot histograms for each selected feature to visualize their distributions
        fig, axes = plt.subplots(
            nrows=(len(selected_features) + 2) // 3,
            ncols=3,
            figsize=(14, 4 * ((len(selected_features) + 2) // 3))
        )
        feature_df.hist(ax=axes.flatten()[:len(selected_features)], edgecolor="k", bins=15)
        plt.suptitle("Distribution of Each Selected Feature", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    

    # ------------------------- Step 8: Scale Data -------------------------------------- #
    # Center and scale the features because we are using models that are sensitive to the variable scales.
    scaler = StandardScaler()
    # Scaling transforms the data so that each feature has a mean of 0 and a standard deviation of 1.
    X_scaled = scaler.fit_transform(feature_df)

    # Use PCA as a visual tool to reduce the dimensionality of the data to 2D for later steps. 
    # PCA helps us capture the most important variance in the data while allowing us to plot it in a way that is easier to interpret.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    
    # ------------------------- Step 9: Hierarchical Clustering ------------------------- #
    # In this tab we will build an unsupervised heirarchical cluserting model which builds a dendrogram of clusters by merging similar observations together. 
    # We use the bottom-up approach where each observation starts alone and then the two most similar clusters are merged at every step until everything is one cluster.

    # Explain the model for user 
    with tab3:
        with st.expander("💡 What is Hierarchical Clustering 💡"):
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

        st.subheader("Hierarchical Clustering Results")
        # Allow users to select a column to label points in the dendrogram for better interpretability. 
        label_col = st.selectbox(
            "Label points in diagram by __________:",
            options=["None"] + df.columns.tolist(),
            index=0
        )

        # use Ward linkage to merges clusters with the smallest increase in total within-cluster variance.
        Z = linkage(X_scaled, method="ward")

        # Plot a  dendrogram to the user two insights:
        ## they can now view similarity in the structures of groups (who merges early).
        ## they can comprehend the reasonable cut heights for k clusters.
        fig, ax = plt.subplots(figsize=(20, 7))
        dendrogram(Z, labels=None, ax=ax)
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Row Label")
        ax.set_ylabel("Distance")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        #Explain these insights to the user
        with st.expander("💡 What is a Dendrogram 💡"):
            st.markdown("""
            A **full dendrogram is the complete merge history of your data.**
            - The bottom **leafs** are all one observation
            - Each **horizontal line** is a merge between two clusters
            - The **height of the line** is the distance between what was merged
            - **Large vertical gap** are cutting where their are the most natural clusters
            """)

        # ------------------------- Step 9b: K & Silhouette Analysis for Heirarchial ------------------------- #
        # Give the user a recommended k based on silhouette scores to help them decide where to cut the dendrogram for optimal clusters.
        # In this step we want to help the user understand how to choose the best number of clusters (k) for their data by 
        # using silhouette scores and PCA visualization to see how groups are formed.

        # create a range of potential k values to evaluate silhouette scores
        k_range = range(2, 11)
        sil_scores = []

        # Loop through each k, fit the Agglomerative Clustering model, and calculate the silhouette score for that k.
        for k in k_range:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            sil_scores.append(score)

        # Calculate the best k
        best_k = list(k_range)[np.argmax(sil_scores)]
        
        # --- Silhouette Diagram ---
        with st.expander("🤍 Hypertune by K 🤍"):
        
        # ------------------------- Step 9c: Hyptertuning with K ------------------------- #
        # Allow the user to select a k value and see how the clusters form in a PCA scatter plot.
        # This helps them understand how different k values affect the cluster structure and visually 
        # confirm the silhouette score recommendations based on how well groups are seperated.
            st.subheader("Hypertune Clusters For Hierarchical Modeling")

            col1, col2 = st.columns([2, 1])
            with col1:
                # add a slider for tunning
                chosen_k = st.slider(
                    "Choose number of clusters (k)",
                    min_value=2, max_value=20, value=best_k,
                )
            with col2:
                # compare the k of choice vs model recommended k
                st.metric("Recommended k", best_k)
                st.metric("Chosen k", chosen_k)

            # use the chosen k to fit the Agglomerative Clustering model and get cluster labels for each point
            chosen_labels = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward").fit_predict(X_scaled)

            # Reduce the data to 2 dimensions for visualization using PCA and color points by their cluster labels 
            # we want to see how well the points are seperated at the chosen k
            fig, ax = plt.subplots(figsize=(10, 7))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=chosen_labels, cmap='viridis',
                                s=60, edgecolor='k', alpha=0.7)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title(f'Agglomerative Clustering @ k={chosen_k}')
            ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.grid(True)

            # if the user selected a column to label points in the diagram label each point with the corresponding value for interpretability
            if label_col != "None":
                for i, label in enumerate(df[label_col].astype(str).values):
                    ax.annotate(label[:4], (X_pca[i, 0], X_pca[i, 1]),
                                fontsize=7, alpha=0.75, xytext=(4, 2),
                                textcoords="offset points")
        
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # add an expander to explain how to interpret the PCA scatter plot 
            with st.expander("💡 PCA Description 💡"):
                st.markdown("""
                This **PCA scatter plot** plots our high-dimensional data into 2D for graph for visualization.
                - Each of these **dot** represents a row in the dataset
                - Each **Color** indicates which cluster it belongs to
                - When there is **Tight, well-separated blobs** this means the data has indicate strong clusters
                - When there is **Overlapping colors** the clusters may not be well defined at this k
                            
                Try a k value that is not the recommended level, and see how they merge!
                """)

        # Add mentrics to compare the silhouette score of the chosen k vs the best k based on silhouette scores 
        # to help users understand how their choice of k impacts quality of the models cluser ability
        with st.expander("🤍 Silhouette Analysis 🤍"):

            # Calculate silhouette scores for the chosen k and the best k to show how well the clusters are formed at each level.
            sample_sil = silhouette_samples(X_scaled, chosen_labels)
            avg_sil = silhouette_score(X_scaled, chosen_labels)

            #calculate the max silhouette score across all k values to show the best possible score for this data 
            max_sil = max(sil_scores)
            max_sil_k = list(k_range)[np.argmax(sil_scores)]

            # diplay the outputs 
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Chosen k", chosen_k)
            col2.metric("Chosen k silhouette score", f"{avg_sil:.3f}")
            col3.metric("Best k", max_sil_k)
            col4.metric("Best silhouette score", f"{max_sil:.3f}")

            # Plot a line graph to show the trend of scores across different k values
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

            # add an expander to explain how to interpret the silhouette analysis and what it means for choosing k
            with st.expander("💡 Silhouette Overview 💡"):
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


    # ------------------------- Step 10: K-Means Modeling  ------------------------- #
    # In this tab we will build a K-Means clustering model which partitions the data into k clusters by starting with a random 
    # center, then iteratively re-assigning points to the nearest cluster centroid and updating the centroids based on the cluster's mean distance
    # We still use PCA to visualize, allow for hypertuning by k, and evaluate the best k with silhouette scores.
    # However it is important to note  the way clusters are formed is different in unsupervised kmeans than hierarchical clustering.
        
        with tab4:
            # provide an overview of the model
            with st.expander("💡 What is K-Means Clustering? 💡"):
                col1, col2 = st.columns([1, 1])
            
                with col1:
                    st.markdown("### What is K-Means?")
                    st.markdown("""
                    K-Means partitions data into **k clusters** by iteratively assigning 
                    points to the nearest cluster centroid, then updating centroids 
                    based on each cluster's mean.
                    
                    This model goes through the **process** of:
                    - **Initialization:** randomly selecting k number of centroids as starting points.
                    - Then **sssignment & updating** means reassigning points to the nearest centroid 
                    and recalculate until the groups come together.
                    
                    Distance between gorups is measured using **Euclidean distance**, and the algorithm 
                    works to minimizes the **Within-Cluster Sum of Squares (WCSS)**.
                    """)

                with col2:
                    st.info("""
                    **Why Use K-Means Clustering?**
                    
                    K-Means shows us the **hidden structure in unlabeled data** 
                    by grouping similar observations together.
                    It can segment data into meaningful subgroups without needing 
                    a target variable and reduce noise within large sets. We use K-Means
                    as a pre-processing tool and to identify patterns for further analysis
                    """)

                    st.info("""
                    **Benefits of the Model**
                    
                    - Fast and simple to implement
                    - Scales well to large datasets
                    - Intuitive and easy to interpret
                    - Useful for customer segmentation, data exploration, and imputation
                    """)

            # ------------------------- Step 10a: allow for hypertuning  ------------------------- #
            st.subheader("K-Means Clustering Results")
            
            # allow users to select a column to label points in the PCA scatter plot for better interpretability.
            label_kmean_col = st.selectbox(
                "Label the points in this diagram by __________:",
                options=["None"] + df.columns.tolist(),
                index=0,
                key="kmeans_label_col"
            )

            # ------------------------- Step 10c: Fit the model and hypertune ------------------------- #
            # Visualization helps us understand how well KMeans has partitioned the data.
            # Since our dataset is high-dimensional, we reduce it to 2D using PCA before plotting.

            with st.expander("🤍 Veiw Results and Hypertune by K 🤍"):
                # Add dynamic user input to select k and see how the clusters form in a PCA scatter plot.
                st.subheader("Hypertune Number of Clusters for K-Means Modeling")

                # let the number of clusters range based on user input 
                ks = range(2, 11)
                wcss = []
                silhouette_scores = []

                # Loop through each k, fit the KMeans model, and calculate both the WCSS and silhouette score for that k 
                # to provide a recommendation for the best k based on silhouette scores.
                for k in ks:
                    #fit the Model with the randome state meaning the initial centroids are randomly selected 
                    # n_init=10 means the algorithm will run 10 times with different random initial centroids
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)

                    # fit the model 
                    km.fit(X_scaled)
                    wcss.append(km.inertia_)
                    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

                best_km_k = list(ks)[np.argmax(silhouette_scores)]

                # add tuning slider
                col1, col2 = st.columns([2, 1])
                with col1:
                    chosen_km_k = st.slider(
                        "Hypertune Number of Clusters for K-Means Modeling",
                        min_value=2, max_value=20, value=best_km_k,
                    )
                with col2:
                    st.metric("Recommended k", best_km_k)
                    st.metric("Your chosen k", chosen_km_k)

                # Set the number of clusters and fit KMeans
                kmeans = KMeans(n_clusters=chosen_km_k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                # Reduce the data to 2 dimensions for visualization using PCA
                fig, ax = plt.subplots(figsize=(10, 7))
                for cluster_label in np.unique(clusters):
                    indices = np.where(clusters == cluster_label)
                    ax.scatter(X_pca[indices, 0], X_pca[indices, 1],
                                alpha=0.7, edgecolor='k', s=60, label=f'Cluster {cluster_label}')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title(f'K-Means Clustering @ k={chosen_km_k}')
                ax.legend(loc='best')
                ax.grid(True)

                # if there is a column selected label each point with the corresponding value for interpretability
                if label_kmean_col != "None":
                    for i, label in enumerate(df[label_kmean_col].astype(str).values):
                        ax.annotate(label[:4], (X_pca[i, 0], X_pca[i, 1]),
                                    fontsize=7, alpha=0.75, xytext=(4, 2),
                                    textcoords="offset points")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # expander for user understanding of the PCA scatter plot and how to interpret the clusters 
                with st.expander("💡 PCA Description 💡"):
                    st.markdown("""
                    This **PCA scatter plot** projects high-dimensional data into 2D for visualization.
                    - Each **dot** represents a row in the dataset
                    - Each **color** indicates which cluster it belongs to
                    - **Tight, well-separated blobs** indicate strong clusters
                    - **Overlapping colors** suggest clusters may not be well defined at this k
                                
                    Try a k value different from the recommendation and see how the clusters shift!
                    """)

                # Expander highlighting the differences between K-Means and Hierarchical clustering 
                with st.expander("💡 How is K-Means Different than Hierarchical? 💡"):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("### How K-Means Works")
                        st.markdown("""
                        K-Means finds clusters by **iterating toward a solution** from a random starting point:
                        
                        The process is:
                        
                        - **Initialization:** Randomly place *k* centroids as initial guesses for cluster centers
                        - **Assignment Step:** Points then are assigned to the nearest centroid 
                        using Euclidean distance
                        - **Update Step:** Centroids are recalculated as the mean of all points 
                        assigned to that cluster
                        - **Iteration:** The process repeats with Assignment and updating until groups 
                        no longer change
                        - **Result:** Outputs *k* clusters whose centroids create the 
                        smallest within-cluster sum of squares (aka inertia)
                        """)


                    with col2:
                        st.markdown("### How Hierarchical is Different")
                        st.markdown("""
                        Hierarchical clustering takes a different approach by 
                        **building structure from the data up** without random initialization:

                        - **No fixed k upfront:** Instead each observation starts as its own cluster 
                        and merges are made step by step
                        - **Merge-based:** Groups are formed by distance between other clusters, 
                        not distance to a centroid
                        - **Shows full history:** The dendrogram preserves every merge decision, 
                        giving you a visual map of how your data is structured at every level
                        """)

                        st.info("""
                        **Key Limitations of K-Means vs Hierarchical**
                        
                        - K-Means works best for **spherical or equally-sized clusters**
                        hierarchical handles **irregular shapes** better
                        - K-Means centroids can be influenced by noisy data points or outliers while 
                        hierarchical is more robust here
                        """)


            # ------------------------- Step 10d: Elbow + Silhouette Analysis ------------------------- #
            # Performance metrics help us evaluate how well the K-Means model has clustered the data and choose the optimal number of clusters (k).

            #highlight the silhouette score of the chosen k vs the best k based on silhouette scores
            with st.expander("🤍 Silhouette & Elbow Analysis 🤍"):
                avg_km_sil = silhouette_score(X_scaled, clusters)
                max_km_sil = max(silhouette_scores)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Chosen k", chosen_km_k)
                col2.metric("Chosen k silhouette score", f"{avg_km_sil:.3f}")
                col3.metric("Best k", best_km_k)
                col4.metric("Best silhouette score", f"{max_km_sil:.3f}")

                # Plot the Elbow Method and Silhouette Score results side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # use the eblow method to plot the WCSS for each k to show how the model's inertia changes as we increase the number of clusters. 
                # inertia is the sum of squared distances of samples to their closest cluster center, so we want to see where the curve starts to flatten out
                # because at this point adding more clusters does not significantly reduce the WCSS, indicating that we have found a good balance
                ax1.plot(list(ks), wcss, marker='o')
                ax1.axvline(chosen_km_k, color="red", linestyle="--", alpha=0.6, label=f"chosen k={chosen_km_k}")
                ax1.axvline(best_km_k, color="green", linestyle="--", alpha=0.6, label=f"best k={best_km_k}")
                ax1.set_xlabel('Number of clusters (k)')
                ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
                ax1.set_title('Elbow Method for Optimal k')
                ax1.legend()
                ax1.grid(True)

                # Plot the Silhouette Score
                ax2.plot(list(ks), silhouette_scores, marker='o', color='green')
                ax2.axvline(chosen_km_k, color="red", linestyle="--", alpha=0.6, label=f"chosen k={chosen_km_k}")
                ax2.axvline(best_km_k, color="green", linestyle="--", alpha=0.6, label=f"best k={best_km_k}")
                ax2.set_xlabel('Number of clusters (k)')
                ax2.set_ylabel('Silhouette Score')
                ax2.set_title('Silhouette Score for Optimal k')
                ax2.legend()
                ax2.grid(True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # and an expander to explain how to interpret the elbow and silhouette charts and use them together for choosing k
                with st.expander("💡 Elbow & Silhouette Overview 💡"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Elbow Method (WCSS)**")
                        st.markdown("""
                        The elbow plot shows the **Within-Cluster Sum of Squares (WCSS)** 
                        against different values of k. It measures each point's distance 
                        to its **own cluster's centroid** meaning how spaced out points of a group are.
                        
                        - The optimal k amound is at the **"elbow"** where improvement slows 
                        and the curve begins to flatten out
                        - We use elbow scores because they are fast to compute, making it a better metric for **large datasets**
                        """)

                    with col2:
                        st.markdown("**Silhouette Score**")
                        st.markdown(f"""
                        The model recommends **k={best_km_k}** (green line) based on the 
                        highest silhouette score of **{max_km_sil:.3f}**.

                        You have chosen **k={chosen_km_k}** (red line), which scores 
                        **{avg_km_sil:.3f}**. See heirarchal section for more on silhouette scores.

                        
                        Use both charts together for the most 
                        robust choice of k.
                        """)


