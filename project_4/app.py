
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
tab1, tab2, tab3 = st.tabs(["🤍Your Dataframe🤍", "🤍Your Features🤍", "🤍 Hierarchical Clustering Model🤍"]) 
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

        label_col = st.selectbox(
            "Label points in diagram by __________:",
            options=["None"] + df.columns.tolist(),
            index=0
        )
            
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

            # ------------------------- Step 8: Elbow + Silhouette Analysis ------------------------- #

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, silhouette_samples
        import matplotlib.cm as cm

        st.subheader("Cluster Count Analysis")


        inertias = []
        sil_scores = []
        k_range = range(2, 20) ##what should I set this as??

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels))

        best_k = list(k_range)[sil_scores.index(max(sil_scores))]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Recommended k (best silhouette)", best_k)
        with col2:
            st.metric("Best silhouette score", f"{max(sil_scores):.3f}")

        # --- Elbow Curve ---
        with st.expander("🤍 Elbow Curve & Silhouette Scores 🤍"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
            axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"k={best_k}")
            axes[0].set_title("Elbow Curve")
            axes[0].set_xlabel("Number of Clusters (k)")
            axes[0].set_ylabel("Inertia")
            axes[0].legend()

            axes[1].bar(list(k_range), sil_scores, color="mediumseagreen", alpha=0.8)
            axes[1].axhline(max(sil_scores), color="darkgreen", linestyle="--", alpha=0.6)
            axes[1].set_title("Silhouette Score vs k")
            axes[1].set_xlabel("Number of Clusters (k)")
            axes[1].set_ylabel("Silhouette Score")
            axes[1].set_xticks(list(k_range))

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = km_best.fit_predict(X_scaled)
            sample_sil = silhouette_samples(X_scaled, cluster_labels)

            fig, ax = plt.subplots(figsize=(10, 6))
            y_lower = 10
            colors = cm.tab10(np.linspace(0, 0.5, best_k))

            for i in range(best_k):
                ith_sil = np.sort(sample_sil[cluster_labels == i])
                size_i = ith_sil.shape[0]
                y_upper = y_lower + size_i
                ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                                alpha=0.7, color=colors[i], label=f"Cluster {i+1}")
                y_lower = y_upper + 10

            avg = np.mean(sample_sil)
            ax.axvline(avg, color="red", linestyle="--", label=f"Avg score: {avg:.3f}")
            ax.set_title(f"Silhouette Diagram — k={best_k}")
            ax.set_xlabel("Silhouette coefficient")
            ax.set_ylabel("Cluster")
            ax.set_yticks([])
            ax.legend(loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # --- Silhouette Diagram ---
        with st.expander("🤍 Hypertune by K 🤍"):
            
        # --- Recommended PCA Plot --- #
            st.subheader("Recommended Clustering")
            st.markdown(f"This is the model's suggested clustering at **k={best_k}** based on the best silhouette score.")

            fig, ax = plt.subplots(figsize=(10, 7))
            plot_colors = plt.cm.tab10.colors

            for cluster_id in range(best_k):
                mask = cluster_labels == cluster_id  # reuse cluster_labels from silhouette section
                ax.scatter(
                    X_pca[mask, 0], X_pca[mask, 1],
                    label=f"Cluster {cluster_id + 1}",
                    color=plot_colors[cluster_id % 10],
                    s=60, alpha=0.8, edgecolors="white", linewidths=0.4
                )

            if label_col != "None":
                for i, label in enumerate(df[label_col].astype(str).values):
                    ax.annotate(label[:4], (X_pca[i, 0], X_pca[i, 1]),
                                fontsize=7, alpha=0.75, xytext=(4, 2),
                                textcoords="offset points")

            ax.set_xlabel(f"First Principal Component ({var_explained[0]*100:.1f}% variance)")
            ax.set_ylabel(f"Second Principal Component ({var_explained[1]*100:.1f}% variance)")
            ax.set_title(f"Model Recommended Clustering — k={best_k}")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


        # ---- NEW: Hypertune k ---- #
            st.subheader("Hypertune Number of Clusters (k)")
            st.markdown("Not happy with the model's recommendation? Choose your own k and see how the clusters change.")

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
            km_chosen = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
            chosen_labels = km_chosen.fit_predict(X_scaled)

            fig, ax = plt.subplots(figsize=(10, 7))
            for cluster_id in range(chosen_k):
                mask = chosen_labels == cluster_id
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                            label=f"Cluster {cluster_id + 1}",
                            color=plot_colors[cluster_id % 10],
                            s=60, alpha=0.8, edgecolors="white", linewidths=0.4)
            if label_col != "None":
                for i, label in enumerate(df[label_col].astype(str).values):
                    ax.annotate(label[:4], (X_pca[i, 0], X_pca[i, 1]),
                                fontsize=7, alpha=0.75, xytext=(4, 2),
                                textcoords="offset points")
            ax.set_xlabel(f"First Principal Component ({var_explained[0]*100:.1f}% variance)")
            ax.set_ylabel(f"Second Principal Component ({var_explained[1]*100:.1f}% variance)")
            ax.set_title(f"PCA Plot — Your Chosen k={chosen_k}")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


        #--------------- max_d Hypertuning ------------------------- #

        with st.expander("🤍 Hypertune by Max Distance 🤍"):
            from scipy.cluster.hierarchy import fcluster

            st.subheader("Hypertune Clusters via max_d Threshold")

            st.markdown(
                "The `max_d` threshold acts as a horizontal cut on the dendrogram — "
                "lower values create more clusters, higher values merge them together."
            )

            # --- Plot number of clusters vs max_d ---
            x_ = []
            y_ = []
            for i in range(1, 11):
                clusters_temp = fcluster(Z, i, criterion='distance')
                x_.append(i)
                y_.append(len(set(clusters_temp)))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x_, y_, marker="o", color="steelblue", linewidth=2)
            ax.set_title("Number of clusters vs max_d threshold")
            ax.set_xlabel("max_d")
            ax.set_ylabel("Number of clusters")
            ax.set_ylim(0, max(y_) + 2)
            ax.set_xlim(1, 10)
            ax.set_xticks(range(1, 11))
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # --- Let user pick max_d ---
            max_d = st.slider(
                "Select max distance threshold",
                min_value=1,
                max_value=10,
                value=2,

            )

            # Compute clusters at chosen max_d
            clusters_maxd = fcluster(Z, max_d, criterion='distance')
            n_clusters_maxd = len(set(clusters_maxd))

            col1, col2 = st.columns(2)
            col1.metric("Chosen max_d", max_d)
            col2.metric("Resulting number of clusters", n_clusters_maxd)



            # --- PCA scatter colored by max_d clusters ---
            fig, ax = plt.subplots(figsize=(10, 7))
            colors = plt.cm.tab10.colors

            for cluster_id in sorted(set(clusters_maxd)):
                mask = clusters_maxd == cluster_id
                ax.scatter(
                    X_pca[mask, 0],
                    X_pca[mask, 1],
                    label=f"Cluster {cluster_id}",
                    color=colors[(cluster_id - 1) % 10],
                    s=60,
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=0.4
                )

            if label_col != "None":
                for i, label in enumerate(df[label_col].astype(str).values):
                    ax.annotate(
                        label[:4],
                        (X_pca[i, 0], X_pca[i, 1]),
                        fontsize=7,
                        alpha=0.75,
                        xytext=(4, 2),
                        textcoords="offset points"
                    )

            ax.set_xlabel(f"First Principal Component ({var_explained[0]*100:.1f}% variance)")
            ax.set_ylabel(f"Second Principal Component ({var_explained[1]*100:.1f}% variance)")
            ax.set_title(f"PCA Plot — max_d={max_d} → {n_clusters_maxd} clusters")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

