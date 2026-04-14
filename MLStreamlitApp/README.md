# Machine Learning Project Overveiw
In traditional programming, humans provide the input and state rules for computers to execute. However, machine learning takes human inputs and outputs (through data and parameter settings) to create the rules that connect the data points. This is done with the final goal of building a computer framework for prediction or decision-making that is applicable to future information.

<p align="center">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/af7b087d-049e-4cf3-9904-bc6dd184c0fa" />
</p>

In this integrated Streamlit dashboard, users can upload their own dataset to train a machine learning model. We allow for the specification of hyperparameters, then evaluate the performance based on metrics that correspond to the model of choice.

<img width="2458" height="1094" alt="image" src="https://github.com/user-attachments/assets/01520cfe-8d73-43e7-910f-01a47ec912b9" />

## Tools Used:
📌Pandas\
📌Seaborn\
📌Matplotlib\
📌Streamlit\
📌Sklearn\
📌graphviz

## The Key Visualizations Used for Evaluation Are:
✅ ***Accuracy: Overall proportion of correct predictions***\
Tells us how well the model performs overall, but does not distinguish between the types of errors being made

✅ Precision: Of all positive predictions, how many were actually correct\
Guards against false positives, but does not think about the positive cases it missed entirely

✅ Recall: Of all actual positives, how many did the model catch\
Guards against false negatives, but it does not think about how many false alarms are made in the process

✅ F-1 Scores: "Harmonic mean" of precision and recall\
The purpose is to "measure if both recall and precision are high, and to ring a bell when one of these two The scores are low" (Grokking).

✅ R^2 Score: Proportion of variance in the target explained by the model

✅ RMSE: The average size of prediction errors

## Interpreting ROC Curves and Confusion Matrix
<img width="1120" height="446" alt="image" src="https://github.com/user-attachments/assets/1840dad5-ed06-457a-9a55-4d23ed263860" />

✅ ***ROC Curces:***\
Plots the rate of true positives versus the false positive rate at various thresholds (the probability cutoff to make a classification)\
The closer the curve hugs the top left corner, the better the model

✅ ***Confusion Matrix:***\
Separates a model's predictions into four categories based on predicted/ actual positives/negatives. A strong model has high numbers along the diagonal and low numbers in the off-diagonal cells (false positives and false negatives). 

<img width="1318" height="1096" alt="image" src="https://github.com/user-attachments/assets/a26d2228-e841-49b3-896b-0462eb1c1ba2" />

✅ ***Residual Plot:***\
Plots the difference between the predicted  and the actual values. The closer the points are scattered randomly around zero, the better the model's fit. However, patterns or curves in the plot indicate that the model does not fully capture the relationship in the data



🖥️ Setup Instructions
- clone the repository
- Install required packages "pip install streamlit pandas seaborn matplotlib"


