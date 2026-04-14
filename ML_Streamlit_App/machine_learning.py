
# ------------------------- Description ------------------------- #
# In this app we will build an interactive streamlit platform for model building and evaluation. 
# The streamlit platform will allow the user to input a data set and select from a menu of machine learning models to build and evaluate.
# We will cover simple linear regression, decision tree classification, and logistic regression. 
# For each model, we will allow the user to select the target variable and feature variables from their uploaded dataset, 
# adjust hyperparameters, and view model performance metrics and visualizations.


# ------------------------- Step 1: Import Libraries ------------------------- #
# Import all necessary libraries for data manipulation, visualization, and machine learning.

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, roc_curve, roc_auc_score, mean_squared_error, f1_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
import graphviz
from sklearn.model_selection import train_test_split


# -- ------------------------- Step 2: Build Streamlit App Layout ------------------------- #
# Add Main Title and Descriptions
t1, t2 = st.columns((1,5)) 
t1.image('images/cartoon-robot-clipart-xl.png', width =200)
t2.title("Exploratory Data Analysis and Machine Learning Models")
t2.markdown(" **Name:** Adelyn Clemmer **| Class:** Intro to Data Science ")
st.text("Upload a CSV or Excel file to get started. Files with size over 10,000KB will cause program to run slow.")

# Allow user to uploead files
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

# Test if file is an excel or csv file and load it into a dataframe
try:
    if uploaded_file is not None:
        # Check file extension and load based on type
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)



# Output an error message if file is not a csv or excel file or if there is an error
except Exception as e:
    st.error(f"Error loading file: {e}")

# Display the users dataframe for visability 
if uploaded_file is not None:
   
    #Drop columns to simplify viewing experience and allow user to focus on relevant features for model building
    st.subheader("🗑️ Drop Columns 🗑️")
    st.text("Select columns to drop from your dataset to focus on relevant features for model building.")
    cols_drop = st.multiselect("Select columns to drop", df.columns.tolist())
    if cols_drop:
        df = df.drop(columns=cols_drop)
    
    # Display the dataframe and some key summary statistics about the data 
    with st.expander("🤍Your Dataframe🤍"):
        st.dataframe(df)
        # Create a side by side layout
        col1, col2, col3 = st.columns(3)

        with col1:
            # display the number of missing data points in a set
            st.markdown("Missing Data")
            missing = df.isnull().sum().reset_index()
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

else:
    # If no file is uploaded, display a warning message prompting the user to upload a better file 
    st.warning("Please upload a CSV or Excel file to begin")

# ---------------------- Steps 3-5: Model Building and Evaluation ------------------------#
# Create an expander to show the model building and evaluation section of the app. 
# This will better our user interface allow the user to focus on the data upload and exploration before moving on to model building.
with st.expander("Machine Learning Models"):
    st.markdown("Once you have uploaded your data, you can select a machine learning model from the sidebar menu to build and evaluate your model. Adjust the hyperparameters to optimize performance and view the model summary for insights into the results.")

    # Create a sidebar menu for model selection
    choice = st.sidebar.selectbox('Menu of Model Types', ['Simple Linear Regression', 'Decision Tree', 'Logistic Regression'] )
    st.sidebar.markdown("*--- * --- * --- * --- * --- * --- * --- * --- *")
    st.sidebar.subheader("⭐ Hyperparameter Tuning ⭐")
    st.sidebar.caption("Adjust these parameter to optimize your model's performance")

    # Create a slider to display different test sizes for the train test split
    test_size = st.sidebar.slider(
        # This has a min value of 0.1 and a max value of 0.5 with a default value of 0.2 and a step of 0.05
        "Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05,
    )

# ---------------------- Steps 3a: Simple Linear Regression Model ------------------------#

   # Simple linear regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables.
   # The goal of simple linear regression is to model the relationship between a target variable and a feature by fitting a linear equation to observed data.
   # The linear regression model assumes that there is a linear relationship between the target variable and the feature variable, and it estimates the coefficients of the linear equation that best fits the data.


    # Create an if statement to display all model types simple linear regression model when the user selects it from the sidebar menu
    if choice == 'Simple Linear Regression':

        st.subheader("Simple Linear Regression")
        st.text("Select the target variable and the feature variable to build a simple linear regression model.")

        # Create two select boxes to allow the user to select the feature variable and the target variable from the dataframe columns
        feature_vars = st.selectbox("Select Feature Variable", df.columns)
        target_vars = st.selectbox("Select Target Variable", df.columns)
        # Create a new dataframe with only the selected feature and target variables and drop any rows with missing values
        X = df[[feature_vars]]
        y = df[target_vars]

        #If the selected variables are not numeric, display an error message prompting the user to select numeric variables
        if not pd.api.types.is_numeric_dtype(df[feature_vars]) or not pd.api.types.is_numeric_dtype(df[target_vars]):
            st.text("Please select numeric variables for both target and feature.")

        # Check if the selected variables are numeric
        if pd.api.types.is_numeric_dtype(df[feature_vars]) and pd.api.types.is_numeric_dtype(df[target_vars]):
            # Split the raw data into training and testing sets by the requested user parameters
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Initialize and train the linear regression model on unscaled data
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = lin_reg.predict(X_test)

            # Plot the regression line
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[feature_vars], y=df[target_vars], color='black', label='Data Points')

            # Use the predict method to get the predicted values for the regression line and plot it
            sns.lineplot(x=df[feature_vars], y=lin_reg.predict(X), color='red', label='Regression Line')
            
            # Set the labels and title of the plot
            plt.xlabel(feature_vars)
            plt.ylabel(target_vars)
            plt.title('Simple Linear Regression')
            plt.legend()
            # Show the line plot
            st.pyplot(plt)

# ---------------------- Steps 3b: Simple Linear Regression Evaluation ------------------------#

            # Create an expander to show the model summary
            with st.expander("🔥Click to view the Model Summary🔥"):
                m1, m2, m3 = st.columns(3)
                
                #Calculate mean squared error
                # The mean squared error measures the average squared difference between the predicted values and the actual values. 
                # A lower mean squared error indicates a better fit of the model to the data.
                mse = mean_squared_error(y_test, y_pred)
                m1.metric("Mean Squared Error", f"{mse:.2f}")

                # Calculate R² score
                # The coefficient of determination, measures the proportion of variance in the target variable that can be explained by the feature variable.
                # An R² score of 1 indicates that the model perfectly explains the variance in the target variable.
                # Am R² score of 0 indicates that the model does not explain any of the variance.
                r2 = lin_reg.score(X_test, y_test)
                m2.metric("R² Score", f"{r2:.2f}")
                
                # Display the coefficients and intercept of the linear regression model
                # The coefficients represent the change in the target variable for a one-unit change in the feature. 
                # The intercept represents the expected value of the target variable when the feature variable is zero
                m3.metric("Intercept", f"{lin_reg.intercept_:.2f}")
                st.markdown("Model Coefficients")
                coef_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Coefficient": lin_reg.coef_
                })
                st.dataframe(coef_df, use_container_width=True)

                # Plot the Residuals
                # Residuals are the differences between the actual values and the predicted values of the target variable.
                # They show how far off the predictions are from the actual values.
                # Points scattered randomly around 0 indicate a good fit.
                st.markdown("Residual Plot")
                fig_res, ax_res = plt.subplots(figsize=(4, 3))
                # take the difference between the actual y and the predicted y to get the residuals
                residuals = y_test - y_pred

                # Create a scatter plot of the predicted values vs the residuals to visualize the residuals
                ax_res.scatter(y_pred, residuals, color='black', alpha=0.5)
                ax_res.axhline(y=0, color='red', linestyle='--')
                ax_res.set_xlabel("Predicted Values")
                ax_res.set_ylabel("Residuals")
                ax_res.set_title("Residual Plot")
                st.pyplot(fig_res, use_container_width=True)
        
    


# ---------------------- Steps 4a: Decision Tree Modeling ------------------------#
# Initalize and train a Decision tree classification which is a supervised machine learning model used for classification tasks.
# We use them becuase they are intuitive and easy to interpret, and they can capture non-linear relationships without needing feature scaling.

    if choice == 'Decision Tree':
        st.subheader("Decision Tree")
        st.text("Select the target variable and feature variables to build a Decision Tree model.")
        
        # Create sliders in the sidebar to allow the user to adjust the hyperparameters of the decision tree model.

        # Max depth is the maximum depth of the tree. It controls how deep the tree can grow. 
        # A deeper tree can capture more complex relationships but risks lead to overfitting which reduces the ability 
        # of the model to generalize to new data.
        max_depth = st.sidebar.slider(
            "Max Depth",
            min_value=1, max_value=20, value=3, step=1,
        )

        # Min samples split is the minimum number of samples required to split an internal node. 
        # It controls how many samples are needed to create a new branch in the tree. When more samples are required to split, the tree will be less complex and less likely to overfit, but it may also underfit if the value is too high.
        # A lower value allows the tree to grow more complex and capture more intricate patterns in the data, 
        # but it also increases the risk of overfitting.
        min_samples_split = st.sidebar.slider(
            "Min Samples Split",
            min_value=2, max_value=20, value=2, step=1,
        )


        # Create select boxes to allow the user to select the target variable and feature variables from the dataframe columns
        # The target variable is the variable we want to predict, and the feature variables are the variables we use to make predictions.
        target_var = st.selectbox("Select Target Variable", df.columns)
        feature_var = st.multiselect("Select Feature Variable", df.columns)


        if feature_var: 
            # Take in the chosen hyperparameters and selected features and target variables
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            # Create a new dataframe with only the selected feature and target variables and drop any rows with missing values
            data_clean = df[feature_var + [target_var]].dropna()

            # Now we use get_dummies to convert the categorical variables into dummy variables which are binary and can be used in a decision tree.
            # Drop_first is used to avoid the dummy variable trap when they are correlated with each other, leading to multicollinearity.
    
            X = pd.get_dummies(data_clean[feature_var], drop_first=True)
            y = data_clean[target_var]

            # Train the model using the selected sample size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Initialize and train the decision tree model on unscaled data
            model.fit(X_train, y_train)
            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Visualize the decision tree to see the structure. This help us understand how the model is making predictions based on the features.
            # This visualization is a benfit of the model because it allows us to see the decision rules and how the features are being used to split the data at each node of the tree.
            st.subheader("Decision Tree Visualization")
            dot_data = tree.export_graphviz(model, feature_names=X_train.columns,
                                        class_names=[str(c) for c in model.classes_],
                                        filled=True)
            graph = graphviz.Source(dot_data)
            st.graphviz_chart(graph)


# ---------------------- Steps 4b: Decision Tree Evaluation ------------------------#
            with st.expander("🔥Click to view the ROC and Model Summary🔥"): 

                # Accuracy is the "overall proportion of correct predictions" made by the model. 
                # Calculate the accuracy of the model by comparing the predicted values to the actual values.
                accuracy = accuracy_score(y_test, y_pred)

                # Percision is the "proportion of positive identifications that were actually correct." 
                # Percision answers the question: "Of the times we predicted a positive, how many were actually positive?"
                # Calculat the precision of the model by comparing the predicted positive values to the actual positive values.
                percision = precision_score(y_test, y_pred, pos_label=model.classes_[1])

                # Recall is the "proportion of actual positives that were identified correctly."
                # Recall answers the question: "Of the times that data was actually positive, how many did we predict as positive?"
                # Calculate the recall of the model by comparing the predicted positive values to the actual positive values
                recall = recall_score(y_test, y_pred, pos_label=model.classes_[1])

                # F1 score is the "harmonic mean of precision and recall." 
                # It is a measure of a model's accuracy that considers both precision and recall.
                f1 = f1_score(y_test, y_pred, pos_label=model.classes_[1])
                
                # Print the metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{accuracy:.2f}")
                m2.metric("Precision", f"{percision:.2f}")
                m3.metric("Recall", f"{recall:.2f}")
                m4.metric("⭐F1 Score⭐", f"{f1:.2f}") 
            
                
                # The ROC curve helps visualize aluate the model's performance across different classification thresholds:
                # It plots the Plots True Positive Rate against False Positive Rate.
                # AUC Summarizes the overall ability of the model to discriminate between classes.
                y_probs = model.predict_proba(X_test)[:, 1]
                # Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
                fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label=model.classes_[1])
                roc_auc = roc_auc_score(y_test, y_probs)
                
                col1, col2 = st.columns(2)

                # Show this information with a confusion matrix
                # A confusion matrix table shows the number of true positives, true negatives, false positives, and false negatives.
                # It helps display the types of errors the model is making.
                with col1:
                    st.markdown("Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))  
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_title('Confusion Matrix')
                    ax_cm.set_xlabel('Predicted') 
                    ax_cm.set_ylabel('Actual')  
                    st.pyplot(fig_cm, use_container_width=True)  
                
                # Compute the AUC score
                with col2:
                    # Plot the ROC curve        
                    # Get the predicted probabilities for the positive class
                    st.markdown("ROC Curve")
                    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))  
                    ax_roc.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    ax_roc.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')  
                    ax_roc.set_xlabel('False Positive Rate') 
                    ax_roc.set_ylabel('True Positive Rate') 
                    ax_roc.set_title('ROC Curve')  
                    ax_roc.legend(loc="lower right")  
                    st.pyplot(fig_roc, use_container_width=True)

    
     
# ---------------------- Steps 5a: Logisitc Regression ------------------------#
# We use logistic regression to model out binary outcomes. 
# The model works by taking the linear combinations of predictors into probabilities in a way that is both
# meaningful and has good interpretability
   
    if choice == 'Logistic Regression':
        # Make titles and descriptors for the interphase
        st.subheader("Logistic Regression")
        st.text("Select the target variable and feature variables to build a Logistic Regression model.")

        #Select particular features and target variable for the model
        target_var_log = st.selectbox("Select Target Variable", df.columns)
        feature_var_log = st.multiselect("Select Feature Variable", df.columns)

        # clean the data by dropping any rows with missing values in the selected feature and target variables
        if feature_var_log:
            data_clean_log = df[feature_var_log + [target_var_log]].dropna()
            X_log = pd.get_dummies(data_clean_log[feature_var_log], drop_first=True)
            y_log = data_clean_log[target_var_log]

            # Split the data into training and testing sets by the requested user parameters
            X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=test_size, random_state=42)

            # Initialize and train the logistic regression model.
            lr_model = LogisticRegression()
            lr_model.fit(X_train_log, y_train_log)

            # Make predictions on the test set
            y_pred_log = lr_model.predict(X_test_log)

# ---------------------- Steps 5b: Logisitc Model Evaluation ------------------------#

            with st.expander("🔥Click to view the ROC and Model Summary🔥"):
            
                # Predicted probabilities for the positive class
                y_probs_lr = lr_model.predict_proba(X_test_log)[:, 1]

                # Accuracy is the overall proportion of correct predictions
                accuracy_log = accuracy_score(y_test_log, y_pred_log)

                # F1 score is the harmonic mean of precision and recall
                f1_log = f1_score(y_test_log, y_pred_log, pos_label=lr_model.classes_[1])

                # Precision is the proportion of positive identifications that were actually correct
                precision_log = precision_score(y_test_log, y_pred_log, pos_label=lr_model.classes_[1])

                # Recall is the proportion of actual positives that were identified correctly
                recall_log = recall_score(y_test_log, y_pred_log, pos_label=lr_model.classes_[1])

                # Calculate the AUC score for the logistic regression model
                roc_auc_log = roc_auc_score(y_test_log, lr_model.predict_proba(X_test_log)[:, 1])

                #Display the metrics in Streamlit
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{accuracy_log:.2f}")
                m2.metric("ROC AUC", f"{roc_auc_log:.2f}")
                m3.metric("⭐F1 Score⭐", f"{f1_log:.2f}")
                m4.metric("Precision", f"{precision_log:.2f}")
                m5.metric("Recall", f"{recall_log:.2f}")

                # ROC curve values and AUC
                fpr_lr, tpr_lr, thresholds_lr = roc_curve(
                    y_test_log, y_probs_lr, pos_label=lr_model.classes_[1]
                )
                roc_auc_lr = roc_auc_score(y_test_log, y_probs_lr)

                col1, col2 = st.columns(2)
                #  Make Streamlit columns
                with col1:  
                    st.markdown("Confusion Matrix")  
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))  
                    
                    # Compute confusion matrix:
                    # Rows are theactual values 
                    # Columns are the predicted values
                    cm = confusion_matrix(y_test_log, y_pred_log)
    
                    # Plot confusion matrix as a heatmap
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        
                    ax_cm.set_title('Confusion Matrix') 
                    ax_cm.set_xlabel('Predicted')     
                    ax_cm.set_ylabel('Actual')           

                    # Display in Streamlit 
                    st.pyplot(fig_cm)  
    
                # Compute the AUC score and visualize ROC curve
                with col2: 
    
                    st.markdown("ROC Curve") 
    
                    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))  
    
                     # Plot ROC curve:
                    ax_roc.plot(fpr_lr, tpr_lr, lw=2, label=f'ROC Curve (AUC = {roc_auc_lr:.2f})')
   
    
                    ax_roc.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')  
                        # Plot diagonal line → baseline model (random guessing)
                        # Helps compare your model vs no-skill model
                        
                    ax_roc.set_xlabel('False Positive Rate')  
                    ax_roc.set_ylabel('True Positive Rate')   
                    ax_roc.set_title('ROC Curve')            
                    
                    # Show legend
                    ax_roc.legend(loc="lower right")  
                       
                    # Display ROC plot in Streamlit
                    st.pyplot(fig_roc, use_container_width=True)
                        
