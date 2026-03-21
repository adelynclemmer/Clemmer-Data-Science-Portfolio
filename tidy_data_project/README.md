# Data Tidying and Visualization 
This project is a data tidying and visualization notebook exploring how federal research and development budgets vary by department as well as alongside overall GDP growth.

<img width="1240" height="670" alt="image" src="https://github.com/user-attachments/assets/f077f7c0-85f5-4976-a86f-17c7702a2649" />


## Tools Used:
📌Pandas\
📌Seaborn\
📌Matplotlib\
📌Streamlit

## Data Sourcing

The data was obtained from a public GitHub data set by jonthegeek and adapted from the Federal Research and Development Spending by Agency set.
You can reference the directory and obtain the data for yourself at the link below:\
[https://github.com/rfordatascience/tidytuesday/tree/main/data/2019/2019-02-12]\

## Overview
The purpose of this project is to analyze the trends in both Total R&D spend over time, GDP growth over time, R&D spend by department over time, and the relative distribution of R&D by department over time. This project uses multiple forms of bar graphs, stacked bar graphs, and line plots to display two main insights:

✅ Even though total GDP and R&D have dramatically increased over time, the relative spend on governmental R&D has dropped from ~5% in 1976 to >1% in 2017\
✅ While no largely dramatic redistributions between government agencies' R&D budget have occurred over the period, there seems to be an increase in focus of social and science-based department R&D spend, like the NIH and the HHS, which grew over the period. Interestingly, the DOD's relative budget has been constrained from 1979 to 2017.    

## Tidy Data Pricipals 
The goal of tidy data is to create a standardized structure. According to Wickham's piece, tidy data is necessary to prepare data for analysis and he presents a clear structure to get to that point. To clean messy data we must ensure\
1. Each variable forms a column
2. Each observation forms a row
3. Each type of observational unit forms a table

🖥️ Setup Instructions
- clone the repository
- Install required packages "pip install streamlit pandas seaborn matplotlib"

