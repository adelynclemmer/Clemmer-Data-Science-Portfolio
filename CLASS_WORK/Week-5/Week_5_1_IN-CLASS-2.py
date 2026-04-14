import pandas as pd            # Library for data manipulation
import seaborn as sns          # Library for statistical plotting
import matplotlib.pyplot as plt  # For creating custom plots
import streamlit as st         # Framework for building interactive web apps

# ================================================================================
#Missing Data & Data Quality Checks
#
# This lecture covers:
# - Data Validation: Checking data types, missing values, and ensuring consistency.
# - Missing Data Handling: Options to drop or impute missing data.
# - Visualization: Using heatmaps and histograms to explore data distribution.
# ================================================================================
st.title("Missing Data & Data Quality Checks")
st.markdown("""
This lecture covers:
- **Data Validation:** Checking data types, missing values, and basic consistency.
- **Missing Data Handling:** Options to drop or impute missing data.
- **Visualization:** Using heatmaps and histograms to understand data distribution.
""")

# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
# Read the Titanic dataset from a CSV file.
df = pd.read_csv("data/titanic.csv")

# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
# Show key statistical measures like mean, standard deviation, etc.
st.write("**Summary Statistics**")
st.dataframe(df.describe())

# ------------------------------------------------------------------------------
# Check for Missing Values
# ------------------------------------------------------------------------------
# Display the count of missing values for each column.
st.write("**Number of Missing Values by Column**")
st.dataframe(df.isnull().sum())

# ------------------------------------------------------------------------------
# Visualize Missing Data
# ------------------------------------------------------------------------------
# Create a heatmap to visually indicate where missing values occur.

st.write("**Heatmap of Missing Values**")
# Create a matplotlib figure and axis for the heatmap.
fig, ax = plt.subplots()
# Create a matplotlib and use figure and axsis for the heatmap

# Plot a heatmap where missing values are highlighted (using the 'viridis' color without a bar)

sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
# Render the heatmap in the Streamlit app.
st.pyplot(fig)


# ================================================================================
# Interactive Missing Data Handling
#
# Users can select a numeric column and choose a method to address missing values.
# Options include:
# - Keeping the data unchanged
# - Dropping rows with missing values
# - Dropping columns if more than 50% of the values are missing
# - Imputing missing values with mean, median, or zero
# ================================================================================

#Let user select a number colun to work with
column = st.selectbox("Choose a column to fill",
df.select_dtypes(include=['number']).columns)

#Provide options for how to handle missing data
method = st.radio("Choose a method", ["Original DF", "Drop Rows", "Drop Columns (>50% Missing)","Impute Mean",
"Impute Median",
"Impute Zero"
])


# Work on a copy of the DataFrame so the original data remains unchanged.
df_clean = df.copy()

# Apply the selected method to handle missing data.
if method == "Original DF":
    pass

elif method == "Drop Rows":
    #Remove all rows that 
    df_clean.dropna(inplace = True)

elif method == "Drop Columns (>50% Missing)":
 #taking the column by name and placing conditions (using brackets for 
 #boolean mask and do it over the {isnull} then take the mean of that to return the percent of 1s and 0s to get to 50%)
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean() > 0.5])

elif method == "Impute Mean":
    #if we want to imput the mean 
    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
elif method == "Impute Median":
    #use fillna to impute median
    df_clean[column] = df_clean[column].fillna(df_clean[column].median())

elif method == "Impute Zero":
    #use fillna to impute 0 over the column
    df_clean[column] =df_clean[column].fillna(0)

st.write(df_clean)

# ------------------------------------------------------------------------------
# Compare Data Distributions: Original vs. Cleaned
# Create two columns in the Streamlit layout for side-by-side comparison.
col1, col2 = st.columns(2)
# --- Original Data Visualization ---
with col1:
    st.subheader("Original Data Distribution")
# Plot a histogram (with a KDE) for the selected column from the original DataFrame.
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True)
    plt.title(f"Original Distribution of {column}")
    st.pyplot(fig)
    st.subheader(f"{column}'s Original Stats")
# Display statistical summary for the selected column.
    st.write(df[column].describe())
#
# Display side-by-side histograms and statistical summaries for the selected column.
# ------------------------------------------------------------------------------
# --- Cleaned Data Visualization ---
with col2:
    st.subheader("Cleaned Data Distribution")
# Plot a histogram (with a KDE) for the selected column from the cleaned DataFrame.
    fig, ax = plt.subplots()
    sns.histplot(df_clean[column], kde=True)
    plt.title(f"Distribution of {column} after {method}")
    st.pyplot(fig)
    st.subheader(f"{column}'s New Stats")
# Display statistical summary for the cleaned data.
    st.write(df_clean[column].describe())

