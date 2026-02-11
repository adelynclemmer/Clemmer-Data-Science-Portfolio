# Core Libraries 
import streamlit as st
import pandas as pd

# Visualization Tools 
import seaborn as sns
import matplotlib.pyplot as plt


#this is the header
 
t1, t2 = st.columns((1,5)) 
t1.image('images/Shopping_cart_with_food_clip_art.svg.png', width =200)
t2.title("Customer Behavior: Income Level's & Product Choice")
t2.markdown(" **Name:** Adelyn Clemmer **| Class:** Intro to Data Science ")

# Load the CSV file
df = pd.read_csv("data/marketing_campaign.csv", sep="\t")
df = df.drop(["ID","NumWebVisitsMonth", "Z_CostContact", "Z_Revenue", "Complain", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "AcceptedCmp1", "AcceptedCmp2", "Response"], axis=1)

# Make Interactive Buttons
st.subheader("Welcome to the Database!")
st.write("First we would like to ask about you:")
st.write("Select the place you shop most:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    online_but=st.button("Online")

with col2:
    instore_but=st.button("In Store")

with col3:
    cat_but=st.button("Catalog")

with col4:
    dis_but=st.button("Discounts")

if online_but:
    st.write("🎉 You clicked Click me!")

elif instore_but:
    st.write("🚀 You clicked Try Me!")

elif cat_but:
    st.write("hhhh")

elif dis_but:
    st.write("yy")



# Insert containers separated into tabs:
data_Overview, customer_Overview_Tab, income_Filter = st.tabs(["Data Overview", "Customer Demography", "Information Visuals"])
customer_Overview_Tab.write("Lets get to know the customers in our data set!")
income_Filter.write("Are you wondering who buys what?")

# You can also use "with" notation:
with data_Overview:
    
       
    
    st.write("Here's our data:")
    st.dataframe(df)

    st.markdown("Here's our summarized data:")
    st.subheader("Summary Statistics")
    st.write(df.describe())



with customer_Overview_Tab:

    st.write("Relationship Status Overview:")
    m1, m2, m3, m4, m5 = st.columns((0.5,1,1,1,1))
    married_count = df[df["Marital_Status"] == "Married"].shape[0]
    single_count = df[df["Marital_Status"] == "Single"].shape[0]
    divorced_count = df[df["Marital_Status"] == "Divorced"].shape[0]
    m1.write('')
    m2.metric(label ='Total Married Couples', value=str(married_count)+ " 💍")
    m3.metric(label ='Total Single Households',value = str(single_count)+ " 😭")
    m4.metric(label = 'Total Divorcee',value = str(single_count)+ " 🚨")
    m1.write('')
    
    
    
    customer_demographic_plot = sns.countplot(x=df["Year_Birth"])
   
    customer_demographic_plot.set_xlabel("Customer Birth Year")
    customer_demographic_plot.set_ylabel("Number of Customers")
    customer_demographic_plot.set_title("Customers by Age")
    
    # Align the Tick Marks
    tick_positions = [3,8,13,18,23,28,33,38,43,48,53]
    tick_labels = ["1890","1900","1910", "1920", "1930", "1940", "1950", "1960", "1970", "1980","1990"]
    customer_demographic_plot.set_xticks(tick_positions)
    customer_demographic_plot.set_xticklabels(tick_labels, rotation=45)
    
    st.pyplot(customer_demographic_plot.get_figure())
    plt.close()
    
    # Filter by education level
    education = st.selectbox("Select an Education Level", df["Education"].unique())
    edu_filtered_df = df[df["Education"] == education]
    st.write(f"Customers with a {education} degree:")
    st.dataframe(edu_filtered_df)

    st.write(f"Now add an income contraint")

with income_Filter:

    income_range = st.slider("Income level:", 1000, 200000, (1000, 200000))
    st.write(income_range)
    r_min, r_max = income_range
    
    

    inc_filtered_df = df[(df["Income"] >= r_min) & (df["Income"] <= r_max)]


    if inc_filtered_df.empty:
        st.warning("No customers match the current filter selection. Try widening the filters.")
        st.stop()

    category_totals = {
        'Wines': inc_filtered_df['MntWines'].sum(),
        'Fruits': inc_filtered_df['MntFruits'].sum(),
        'Meat Products': inc_filtered_df['MntMeatProducts'].sum(),
        'Fish Products': inc_filtered_df['MntFishProducts'].sum(),
        'Sweet Products': inc_filtered_df['MntSweetProducts'].sum(),
        'Gold Products': inc_filtered_df['MntGoldProds'].sum()
    }

    

    # Create a dataframe for plotting
    category_df = pd.DataFrame(list(category_totals.items()), 
                               columns=['Category', 'Total Amount'])
    
    # Create the bar plot
    #fig, ax = plt.subplots(figsize=(10, 6))
    income_bar = plt.bar(category_df['Category'], category_df['Total Amount'])

    
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Total Purchase Amount ($)', fontsize=12)
    plt.title(f'Purchase Distribution (Income: ${r_min:,} - ${r_max:,})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(plt.gcf())
    plt.close()
    st.dataframe(inc_filtered_df)


    








