# Customer Behavior: Income Levels & Product Choice

Using data from Kaggle, this dashboard allows users to explore customer purchasing patterns   
through demographic filters and income-based analysis. The dataset was sourced from a marketing  
study aimed at uncovering "a detailed analysis of a company's ideal customers," and we will leverage  
these features (income level, education, age, and relationship status) to analyze the impact on product  
spending across six categories.

<p align="center">
<img height="350" alt="image" src="YOUR_CENTERED_SCREENSHOT_HERE" />
</p>

Begin with a fun shopping prompt before exploring the data through summary statistics, demographic overviews, 
and a dynamic income filter that updates product spending charts in real time!

<img alt="image" src="YOUR_FULL_WIDTH_SCREENSHOT_HERE" />

## Tools Used:
📌 Pandas\
📌 Seaborn\
📌 Matplotlib\
📌 Streamlit

## The Key Features of the Dashboard Are:
✅ *** Shopping Prompt:***\
Users select how they prefer to shop and receive a personalized message alongside a live count 
of how many times that method appears in the dataset!

✅ ***Data Overview Tab:***\
Displays the full cleaned dataframe alongside summary statistics using describe() so users 
can quickly understand the structure and distribution of the data

✅ ***Customer Demography Tab:***\
Breaks down relationship status with metric cards, and visualizes the age distribution of customers 
using a Seaborn count plot

✅ ***Income & Purchase Behavior Tab:***\
A dynamic income range slider filters customers by income level and recalculates total spending 
across six product categories: Wines, Fruits, Meat, Fish, Sweets, and Gold Products all while updating as 
a live updating bar chart

## ✅ Compliments Portfolio ✅
This project was my introduction into building user-driven data exploration tools! I learned to connect 
interactive filters directly to live visual EDA outputs. My goal this semester is to work on projects that are
applicable and insightful to my interests outside DS. Working with real marketing data strengthened my understanding of how demographic 
variables like income and education shape consumer behavior. This project merged my business interest with
data-driven conclusions which I will focus future projects on as well.

## 🖥️ Setup Instructions
- Clone the repository
- Install required packages: `pip install streamlit pandas seaborn matplotlib`
- Run the app: `streamlit run app.py`
- Data is loaded automatically from the included CSV file
