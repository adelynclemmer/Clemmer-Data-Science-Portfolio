# Unsupervised Machine Learning Project Overview
Check it out at this link:https://dgx3ggwexy9v7s34kafdnf.streamlit.app

Unsupervised machine learning works with **unlabeled data** where there is no target variable to predict. 
Instead, our goal when using these tools is to identify hidden structures, find patterns, and
group similar observations to uncover insights or aid in processing down the line! 
This unsupervised machine learning app allows users to begin learning about two of the most popular unsupervised 
clustering models on the data set they choose! The tool pays close attention to K (the number of clusters and offers insight 
into the impact of grouping across models and datasets.

<p align="center">
<img height="400" alt="image" src="https://github.com/user-attachments/assets/34d94c4b-7f8e-4620-8e4d-327b640952e6" />
<img height="350" alt="image" src="https://github.com/user-attachments/assets/3bbf52bb-f172-4bd4-9b20-4c247c043aa2" />
</p>


In this Streamlit dashboard, users can upload their own dataset to run both 
a Hierarchical and a K-Means clustering model. The app allows for hyperparameter tuning 
through tools to control cluster numbers. The output also evaluates cluster quality using 
metrics that help the user identify the most natural groupings in their data.


## Tools Used:
📌 Pandas\
📌 Numpy\
📌 Matplotlib\
📌 Streamlit\
📌 Scipy\
📌 Sklearn (StandardScaler) (PCA) (KMeans & AgglomerativeClustering\
📌 Sklearn Metrics (silhouette_score & silhouette_samples)

## Two Featured Models:
✅ ***Hierarchical (Agglomerative) Clustering:***\
This unsupervised model builds clusters from the bottom up! Each observation starts as its own group, then 
the most similar clusters merge at every step until everything is one group. We can see the process visualized
through our diagram. Check out Streamlit to learn more!

✅ ***K-Means Clustering:***\
The K-Means model splits data into clusters by iterating to assign each data point to the nearest 
centroid (middle of the group), then recalculating centroids as the mean of the cluster. While K-Means requires 
k to be specified before the model runs, it has the benefit of being great at classification tasks, especially if they 
are binary. Try it out!

## Key Visualizations Used for Evaluation:

<p align="center">
<img height="350" alt="image" src="https://github.com/user-attachments/assets/c0638a78-81da-4692-8038-97d54aae0f7b" />
</p>

✅ ***Dendrogram:***\
A visual of the complete merge history of your data. Each leaf in your dendrogram is an observation, each horizontal 
line is a merge, and the height of the line represents the distance between what was joined. You can observe when large 
vertical gaps reveal where the most natural cluster cuts exist.

✅ ***PCA Scatter Plot:***\
This visual compresses high-dimensional data into 2D. Each dot is a row in the dataset 
and each color indicates cluster group. We are looking for tight, well-separated blobs which mean strong 
clusters. However,  overlapping colors suggest the chosen k may not be well defined.

✅ ***Elbow Method (WCSS):***\
In this performance metric, we measure the Within-Cluster Sum of Squares against different values of k. 
The optimal k sits at the "elbow" where improvement begins slows. Use this for speed and for large datasets.

✅ ***Silhouette Score:***\
Measures how similar each point is to its own cluster versus the nearest neighboring cluster. 
Our platform will ensure the use of the silhouette score alongside the elbow method when possible. 

## 🖥️ Setup Instructions
- Clone the repository
- Install required packages: `pip install streamlit pandas numpy matplotlib scikit-learn scipy`
- Run the app: `streamlit run app.py`
- Upload a CSV or Excel file to get started
