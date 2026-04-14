 #mport neccesary libraries
import pandas as pd
import numpy as np

# Download data from Kaggle (conda install kagglehub)
import kagglehub
# This is the easiest way to upload a kaggle file into your environment 

# Download latest version
path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
print("Path to dataset files:", path)

# Import dataframe
df = pd.read_csv(f"{path}/Student_Performance.csv")
df