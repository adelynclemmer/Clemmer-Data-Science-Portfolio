# Core Libraries 
import streamlit as st
import pandas as pd

# Visualization Tools 
import seaborn as sns
import matplotlib.pyplot as plt
 
 # Create the header for the app 
t1, t2 = st.columns((1,5)) 
t1.image('images/Shopping_cart_with_food_clip_art.svg.png', width =200)
t2.title("Customer Behavior: Income Level's & Product Choice")
t2.markdown(" **Name:** Adelyn Clemmer **| Class:** Intro to Data Science ")

 # draft description
st.write("Using data from Kaggle, this dashboard allows you to visualize the set through summary statistics and an age bar chart. You can then explore customer purchasing patterns by using filters to determine popular purchase goods by income level. Navitgate to each section of the dashboard using the tabs below.")


# Load the CSV file
df = pd.read_csv("data/marketing_campaign.csv", sep="\t")
df = df.drop(["ID","NumWebVisitsMonth", "Z_CostContact", "Z_Revenue", "Complain", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "AcceptedCmp1", "AcceptedCmp2", "Response"], axis=1)

st.subheader("Welcome to the Database!")
st.write("First we would like to ask about you:")
st.write("Select the place you shop most:")

# Make Interactive Buttons
# Use columns for size and positioning on the buttons

col1, col2, col3, col4 = st.columns(4)

with col1:
    online_but=st.button("Online")

with col2:
    instore_but=st.button("In Store")

with col3:
    cat_but=st.button("Catalog")

with col4:
    dis_but=st.button("Discounts")


# Use if and elif statements to display personlized messages to the use and a variable counter

if online_but:
    wb_count = df["NumWebPurchases"].sum()
    st.write("🎉 Online is a great place to shop 🎉")
    st.write(f"People in this data set have also shopped online {wb_count} times!")
   
elif instore_but:
    is_count = df["NumStorePurchases"].sum()
    st.write("🛍 In the story is such a classy way to shop 🛍")
    st.write(f"You're not alone... others have shopped in store {is_count} times according to this data set!")

elif cat_but:
    cat_count = df["NumCatalogPurchases"].sum()
    st.write("📦 Catalogs are great ways to shop 📦")
    st.write(f"Others agree with you! There are {cat_count} instances of cataglog purchasing in this data!")


elif dis_but:
    dis_count = df["NumDealsPurchases"].sum()
    st.write("🚫 Get those deals! What a great way to shop! 🚫")
    st.write(f"People have made discount purchases {dis_count} times in this data!")




# Insert containers separated into tabs:
data_Overview, customer_Overview_Tab, income_Filter = st.tabs(["Data Overview", "Customer Demography", "Information Visuals"])
customer_Overview_Tab.write("Lets get to know the customers in our data set!")

# Make the three tabs seperating the dashboard

with data_Overview:
    st.subheader("Here's our data:")
    st.dataframe(df)
    st.subheader("Summary Statistics:")
    # use describe() to show a quantitative summary of information
    st.write(df.describe())


with customer_Overview_Tab:
    st.subheader("Relationship Status Overview:")
    # space out relational categories on the dashboard and filter the data frame for a matching value, use shape to get the number of rows
    m1, m2, m3, m4, m5 = st.columns((0.5,1,1,1,1))
    married_count = df[df["Marital_Status"] == "Married"].shape[0]
    single_count = df[df["Marital_Status"] == "Single"].shape[0]
    divorced_count = df[df["Marital_Status"] == "Divorced"].shape[0]
    m1.write('')
    m2.metric(label ='Total Married Couples', value=str(married_count)+ " 💍")
    m3.metric(label ='Total Single Households',value = str(single_count)+ " 😭")
    m4.metric(label = 'Total Divorcee',value = str(single_count)+ " 🚨")
    m1.write('')
    
    st.subheader("Age Demographics:")
    # make a count plot to display the amount of each birth year in the data
    customer_demographic_plot = sns.countplot(x=df["Year_Birth"])
    # create graph labels
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
    st.subheader("Education Filter:")
    education = st.selectbox("Select an Education Level", df["Education"].unique())
    edu_filtered_df = df[df["Education"] == education]
    st.write(f"Customers with a {education} degree:")
    st.dataframe(edu_filtered_df)


with income_Filter:
    st.subheader("Income and Purchase Behavior")
    
    # Create a slider for users to filter customers by income range
    income_range = st.slider("Income level:", 1000, 200000, (1000, 200000))
    
    # Split the income range into minimum and maximum values
    r_min, r_max = income_range
    
    # Filter the dataframe to only include customers within the selected income range
    inc_filtered_df = df[(df["Income"] >= r_min) & (df["Income"] <= r_max)]

    # Check if any customers match, show a warning and stop if false
    if inc_filtered_df.empty:
        st.warning("No customers match the current filter selection. Try widening the filters.")
        st.stop()

    # Calculate the total amount spent on each product category for filtered customers
    category_totals = {
        'Wines': inc_filtered_df['MntWines'].sum(),
        'Fruits': inc_filtered_df['MntFruits'].sum(),
        'Meat Products': inc_filtered_df['MntMeatProducts'].sum(),
        'Fish Products': inc_filtered_df['MntFishProducts'].sum(),
        'Sweet Products': inc_filtered_df['MntSweetProducts'].sum(),
        'Gold Products': inc_filtered_df['MntGoldProds'].sum()
    }


    # Create a dataframe for plotting
    category_df = pd.DataFrame(list(category_totals.items()),  columns=['Category', 'Total Amount'])
    
    # Create the bar plot
    income_bar = plt.bar(category_df['Category'], category_df['Total Amount'])

    # Customize the chart labels and title
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Total Purchase Amount ($)', fontsize=12)
    plt.title(f'Purchase Distribution (Income: ${r_min:,} - ${r_max:,})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # adjust the spacing so no label is cut off
    plt.tight_layout()

    # Display the chart in Streamlit
    st.pyplot(plt.gcf())
    plt.close()
    st.subheader("Here's the full filtered data:")

    st.dataframe(inc_filtered_df)


    








