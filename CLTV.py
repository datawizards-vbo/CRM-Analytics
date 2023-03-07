# Customer Lifetime Value

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Setting display options for pandas
pd.set_option('display.max_columns', None) # To display all columns
pd.set_option('display.float_format', lambda x: '%.5f' % x) # To format floating point numbers

# Step 1.
# Prepare the data

# Reading the Excel file using pandas
df = pd.read_excel(r"CRM_Analytics\online_retail_II.xlsx", sheet_name="Year 2009-2010")

# Printing the first five rows of the dataframe
df.head()

# Checking for null values in the dataframe
df.isnull().sum()

# Removing rows where the Invoice column contains "C"
df = df[~df["Invoice"].str.contains("C", na=False)]

# Computing descriptive statistics for the numerical columns
df.describe().T

# Removing rows with non-positive quantity values
df = df[(df['Quantity'] > 0)]

# Dropping rows with null values
df.dropna(inplace=True)

# Adding a new column "TotalPrice" by multiplying "Quantity" and "Price" columns
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Grouping the dataframe by "Customer ID" and computing aggregate statistics
cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

# Renaming the columns of the new dataframe
cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']


# Step 2. 
# Average Purchase Value = total_purchase / total_transaction

# Computing and adding "average_purchase_value" to the dataframe
cltv_c["average_order_value"] = cltv_c["total_purchase"] / cltv_c["total_transaction"]

# Step 3. 
# Average Purchase Frequency = total_transaction / number_of_unique_customers

# Computing and adding "average_purchase_frequency" to the dataframe
cltv_c["average_purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]


# Step 4. 
# Repeat Rate & Churn Rate 
# Repeat Rate = number_of_customers_making_more_than_one_purchase / number_of_unique_customers

# Computing the repeat rate and churn rate
repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate


# Step 5. 
# Profit Margin =  total_purchase * 0.10

# Computing and adding a new column "profit_margin" to the dataframe
cltv_c['profit_margin'] = cltv_c['total_purchase'] * 0.10


# Step 6. 
# Customer Value = average_purchase_value * average_purchase_frequency


# Computing and adding a new column "customer_value" to the dataframe
cltv_c['customer_value'] = cltv_c['average_purchase_value'] * cltv_c["average_purchase_frequency"]


# Step 7. 
# Customer Lifetime Value 
# CLTV = (customer_value / churn_rate) * profit_margin)

# Computing and adding a new column "cltv" to the dataframe
cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

# Sorting the dataframe by CLTV in descending order and displaying the top 5 rows
cltv_c.sort_values(by="cltv", ascending=False).head()

# Step 8.
# Creating Segments

# create segments based on CLTV score quartiles
cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

# sort the DataFrame by CLTV in descending order again
cltv_c.sort_values(by="cltv", ascending=False).head()

# group the DataFrame by segment and aggregate statistics
cltv_c.groupby("segment").agg({"count", "mean", "sum"})

# save the DataFrame to a CSV file
cltv_c.to_csv("cltv_c.csv")

# Step 9.
# Functionalizing all steps

def create_cltv_c(dataframe, profit=0.10):
    """
    This function takes a Pandas DataFrame containing transaction data and calculates customer lifetime value (CLTV) metrics
    for each customer. The function returns a new DataFrame containing the CLTV metrics and a segment label for each customer.

    Parameters:
        dataframe (pandas.DataFrame): A DataFrame containing transaction data, with columns for customer ID, invoice ID, quantity,
                                       price, and date.
        profit (float): The profit margin to use in CLTV calculations. Default is 0.10 (10%).

    Returns:
        cltv_c (pandas.DataFrame): A DataFrame containing customer lifetime value (CLTV) metrics and a segment label for each customer.

    """
    # Data preparation
    # Remove cancelled orders and negative quantity items
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe.dropna(inplace=True)

    # Calculate TotalPrice for each item
    dataframe["Total_Purchase"] = dataframe["Quantity"] * dataframe["Price"]

    # Group data by customer ID and calculate total transactions, units, and revenue
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'Total_Purchase': lambda x: x.sum()})

    # Rename the columns for clarity
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_purchase']

    # Calculate average order value
    cltv_c['avg_purchase_value'] = cltv_c['total_purchase'] / cltv_c['total_transaction']

    # Calculate purchase frequency
    cltv_c["avg_purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]

    # Calculate repeat rate and churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # Calculate profit margin
    cltv_c['profit_margin'] = cltv_c['total_purchase'] * profit

    # Calculate customer value
    cltv_c['customer_value'] = (cltv_c['avg_purchase_value'] * cltv_c["avg_purchase_frequency"])

    # Calculate customer lifetime value
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

    # Segment customers based on their CLTV scores
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    # Return the new DataFrame with CLTV metrics and segment labels
    return cltv_c


# Call the create_cltv_c function to calculate CLTV metrics for each customer
clv = create_cltv_c(df)



