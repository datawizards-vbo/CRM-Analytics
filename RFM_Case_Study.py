# Customer Segmentation with RFM Analysis

# TASK 1. Data Understanding

# Import necessarily libraries
import pandas as pd
import datetime as dt

# Set pandas options to display all columns and rows, float format, and comment out display width
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 1. Read the flo_data_20K.csv file and create a copy of the dataframe
df_ = pd.read_csv(r"CRM_Analytics\flo_data_20k.csv")
df = df_.copy()
df.head()

# 2. Analyze the data set including the first 10 observations, variable names, dimensions, descriptive statistics, 
# missing values, and variable types
df.head(10)
df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()

# 3. Create new variables to show the total number of orders and the total customer value for each customer 
# who shops from both online and offline platforms
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 4. Analyze variable types and convert date columns to datetime
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# 5. Analyze the distribution of the number of customers, total number of items purchased, and total customer 
# value in each order channel
df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"})

# 6. Sort and display the top 10 customers who bring the highest customer value 
df.sort_values("customer_value_total", ascending=False)[:10]

# 7. Sort and display the top 10 customers who have the highest number of orders
df.sort_values("order_num_total", ascending=False)[:10]

# 8. Function to prepare the data
def data_prep(dataframe):
    # Create new variables to show the total number of orders and the total customer value for each customer 
    # who shops from both online and offline platforms
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    # Convert date columns to datetime
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df

# TASK 2: Calculate RFM Metrics


# Define the analysis date as 2 days after the date of the latest order in the dataset
latest_order_date = df["last_order_date"].max()
analysis_date = latest_order_date + dt.timedelta(days=2) # add 2 days to the latest order date

# Create a new dataframe to store customer_id, recency, frequency, and monetary values for RFM analysis
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"] # Add customer IDs from original dataframe
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]') # Calculate recency as the number of days between the analysis date and the date of the customer's last order
rfm["frequency"] = df["order_num_total"] # Add total number of orders for each customer from the original dataframe
rfm["monetary"] = df["customer_value_total"] # Add total monetary value of orders for each customer from the original dataframe

# Print the first five rows of the RFM dataframe to check that it has been created correctly
print(rfm.head())


# TASK 3: Calculate RF and RFM Scores

# Convert the recency, frequency, and monetary metrics into scores from 1 to 5 using qcut, and save them as separate columns in the rfm dataframe
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1]) # Divide recency values into 5 equal-sized bins and assign labels of 5 (highest) to 1 (lowest) to each bin
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]) # Divide frequency values into 5 equal-sized bins based on their rank, and assign labels of 1 (lowest) to 5 (highest) to each bin
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5]) # Divide monetary values into 5 equal-sized bins and assign labels of 1 (lowest) to 5 (highest) to each bin

# Combine the recency_score and frequency_score columns into a single column called RF_SCORE
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)) # Concatenate the recency_score and frequency_score columns into a single column

# Combine the recency_score, frequency_score, and monetary_score columns into a single column called RFM_SCORE
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str)) # Concatenate all three score columns into a single column


# TASK 4: Define RF Scores as Segments
# Define segments for more meaningful interpretation of RFM scores and convert RF_SCORE to segments using seg_map
seg_map = {
    r'[1-2][1-2]': 'hibernating', # Customers who have low recency, frequency, and monetary value
    r'[1-2][3-4]': 'at_Risk', # Customers who have low recency but higher frequency and monetary value
    r'[1-2]5': 'cant_loose', # Customers who have high monetary value but low recency and frequency
    r'3[1-2]': 'about_to_sleep', # Customers who have low recency and frequency but higher monetary value
    r'33': 'need_attention', # Customers who have low recency, frequency, and monetary value
    r'[3-4][4-5]': 'loyal_customers', # Customers who have high recency and frequency but lower monetary value
    r'41': 'promising', # Customers who have high recency, low frequency, and monetary value
    r'51': 'new_customers', # Customers who have high frequency but low recency and monetary value
    r'[4-5][2-3]': 'potential_loyalists', # Customers who have high recency and monetary value but lower frequency
    r'5[4-5]': 'champions' # Customers who have high recency, frequency, and monetary value
}

# Replace RF_SCORE with corresponding segment using seg_map and add a new 'segment' column to the rfm DataFrame
rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

# TASK 5. Action Time!
# 1. Analyze the recency, frequency, and monetary averages of the segments.
# Group by the "segment", and aggregate the means and counts of the "recency", "frequency", and "monetary" columns.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# 2. Using RFM analysis, find the customers in the relevant profile for 2 cases and save their customer IDs to CSV files.

# a. FLO is introducing a new women's shoe brand. The prices of the brand's products are above the general customer preferences. Therefore, it is desired to communicate
# with the customers in the relevant profile who will be interested in the promotion and sales of the brand. These customers are planned to be loyal and
# those who shop from the women's category. Save the customer IDs to a CSV file named "new_brand_target_customer_ids.csv".
target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("new_brand_target_customer_ids.csv", index=False)
print(cust_ids.shape)

# b. A discount of nearly 40% is planned for men's and children's products. Previously good customers who are interested in the discounted categories but have not shopped for a long time, and new customers are wanted to be targeted specifically. Save the customer IDs to a CSV file named "discount_target_customer_ids.csv".
target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("discount_target_customer_ids.csv", index=False)
print(cust_ids.shape)


# BONUS. Functionalize all steps

def create_rfm(dataframe):
    # Data Preparation
    # Create a new column for the total number of orders
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    # Create a new column for the total customer value
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    # Get the columns that contain dates and convert them to datetime format
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Calculate RFM Metrics
    # Determine the latest order date and set the analysis date to the next day
    latest_order_date = dataframe["last_order_date"].max()
    analysis_date = latest_order_date + dt.timedelta(days=1)
    # Create a new empty dataframe to store RFM values
    rfm = pd.DataFrame()
    # Add customer ID, recency, frequency, and monetary columns to the RFM dataframe
    rfm["customer_id"] = dataframe["master_id"]
    rfm["recency"] = (analysis_date - dataframe["last_order_date"]).astype('timedelta64[D]')
    rfm["frequency"] = dataframe["order_num_total"]
    rfm["monetary"] = dataframe["customer_value_total"]

    # Calculate RF and RFM Scores
    # Divide recency, frequency, and monetary into 5 equal-sized groups and assign scores
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    # Combine recency and frequency scores to create RF Score
    rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    # Combine RF Score and monetary score to create RFM Score
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

    # Map RF Scores to segment names
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    # Map RF Scores to segment names and add the segment column to the RFM dataframe
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    # Return the RFM dataframe 
    return rfm[["customer_id", "recency","frequency","monetary","RF_SCORE","RFM_SCORE","segment"]]

rfm_df = create_rfm(df)




