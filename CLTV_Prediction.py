# Prediction of Customer Lifetime Value

# Import necessary libraries
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

# Set display options for pandas dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# Step 1.
# Data Preprocessing

# Read the data from the excel file
df = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")

# Print the first 5 rows of the dataframe
df.head()

# Check for missing values in the dataframe
df.isnull().sum()

# Print descriptive statistics of the dataframe
df.describe().T

# Drop rows with missing values
df.dropna(inplace=True)

# Remove rows with negative or zero quantity and cancelled orders
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# Define a function to calculate outlier limits for a variable in a dataframe
def outlier_thresholds(dataframe, variable):
    # Calculate the 1st and 99th percentile of the variable
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    # Calculate the interquartile range of the variable
    interquantile_range = quartile3 - quartile1
    # Calculate the upper and lower limits for outliers
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    # Return the lower and upper limits
    return low_limit, up_limit

# Define a function to replace outlier values with upper and lower limits
def replace_with_thresholds(dataframe, variable):
    # Get the lower and upper limits for the variable
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # Replace values below the lower limit with the lower limit
    #dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    # Replace values above the upper limit with the upper limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Replace outlier values with threshold limits
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# Add a new column for total price of each order
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Set today's date as 11 December 2011
today_date = dt.datetime(2011, 12, 11)

# Step 2.
# Preparing the Lifetime Value Data Structure


# Calculate the recency, T, frequency and monetary values for each customer
cltv_df = df.groupby('Customer ID').agg({
    'InvoiceDate': [
        lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,  # recency
        lambda InvoiceDate: (today_date - InvoiceDate.min()).days  # T
    ],
    'Invoice': lambda Invoice: Invoice.nunique(),  # frequency
    'TotalPrice': lambda TotalPrice: TotalPrice.sum()  # monetary
})

# Drop the first level of column names
cltv_df.columns = cltv_df.columns.droplevel(0)

# Rename the columns to more descriptive names
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# Calculate the average monetary value per purchase
cltv_df["avg_monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# Remove customers with a frequency of less than 2 (i.e. customers who made only one purchase)
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# Convert recency and T values from days to weeks
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7


# Step 3.
# BG-NBD Model

# Create a BetaGeoFitter object with penalizer_coef of 0.001
bgf = BetaGeoFitter(penalizer_coef=0.001)

# Fit the BG-NBD model to the data
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# Find the top 10 customers with the highest expected number of purchases in the next week
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

# Predict the number of purchases for the next week for all customers and find the top 10 with the highest expected number of purchases
bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

# Add a column to the data frame with the expected number of purchases for the next week for each customer
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

# Find the top 10 customers with the highest expected number of purchases in the next month
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

# Add a column to the data frame with the expected number of purchases for the next month for each customer
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# Find the expected number of purchases for the next 3 months for all customers
bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

# Add a column to the data frame with the expected number of purchases for the next 3 months for each customer
cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# Plot the number of transactions in each time period
plot_period_transactions(bgf)

# Display the plot
plt.show()

# Step 4.
# Gamma-Gamma Submodel

# Instantiate a GammaGammaFitter model with a penalizer coefficient of 0.01
ggf = GammaGammaFitter(penalizer_coef=0.01)

# Fit the model using the frequency and monetary columns of the cltv_df dataframe
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# Calculate the conditional expected average profit for the top 10 customers in the dataset
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

# Calculate the conditional expected average profit for all customers in the dataset, sorted in descending order
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

# Add a new column to the cltv_df dataframe that contains the expected average profit for each customer
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

# Sort the cltv_df dataframe by the expected average profit column in descending order and show the top 10 customers
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

# Step 5.
# Calculation of CLTV with BG-NBD and GG models

# Create BG-NBD and GG models
bgf = BetaGeoFitter(penalizer_coef=0.001)
ggf = GammaGammaFitter(penalizer_coef=0.01)

# Fit models to the data
bgf.fit(cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# Calculate CLTV using models
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 months
                                   freq="W",  # frequency of T
                                   discount_rate=0.01)

# Show the first 5 rows of CLTV
cltv.head()

# Reset the index of CLTV dataframe
cltv = cltv.reset_index()

# Merge CLTV with the original dataset
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

# Sort customers by CLTV in descending order and show the top 10
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Step 6.
# Creating segments based on CLTV

# Create segments based on CLTV and add them as a new column to the dataframe
cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

# Sort customers by CLTV in descending order and show the top 50 with their segments
cltv_final.sort_values(by="clv", ascending=False).head(50)

# Group customers by their segments and show count, mean, and sum values for each segment
cltv_final.groupby("segment").agg({"count", "mean", "sum"})

# Step 6.
# Functionalize all steps

# Define the function that will create the CLTV prediction
def create_cltv_p(dataframe, month=3):
    """
    This function takes a transactional dataframe as input and returns a dataframe with CLTV predictions.

    Parameters:
    dataframe: transactional dataframe
    month: int, optional (default: 3), the time period in months to calculate the CLTV for.

    Returns:
    cltv_final: dataframe with CLTV predictions
    """
    # 1. Data Preprocessing
    # Drop the missing values from the dataset
    dataframe.dropna(inplace=True)
    # Remove the rows containing "C" in the "Invoice" column
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    # Remove the rows with negative or zero "Quantity" values
    dataframe = dataframe[dataframe["Quantity"] > 0]
    # Remove the rows with negative or zero "Price" values
    dataframe = dataframe[dataframe["Price"] > 0]
    # Replace the outliers in "Quantity" and "Price" columns with the appropriate thresholds
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    # Add a new column named "TotalPrice" by multiplying "Quantity" and "Price" columns
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    # Set the date for reference as the last purchase date in the dataset
    today_date = dt.datetime(2011, 12, 11)

    # Group the cleaned dataset by "Customer ID" and calculate the required metrics for each customer
    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    # Clean the column names of the resulting dataframe
    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    # Calculate the average transaction value for each customer
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    # Remove the customers with only one purchase from the dataset
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    # Convert "recency" and "T" columns from day-based values to week-based values
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. Build the BG-NBD Model
    # Create an instance of the BetaGeoFitter class
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    # Fit the model to the "frequency", "recency", and "T" columns of the dataset
    bgf.fit(cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])
    # Predict the expected number of purchases for each customer in the next 1 week, 1 month, and 3 months
    cltv_df["expected_purc_1_week"] = bgf.predict(1, cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])
    cltv_df["expected_purc_1_month"] = bgf.predict(4, cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])
    cltv_df["expected_purc_3_month"] = bgf.predict(12, cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])

    # 3. Build the Gamma Gamma Submodel
    # Initialize Gamma-Gamma Fitter and fit it to the data
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(dataframe['frequency'], dataframe['monetary'])
    
    # Calculate expected average profit using the fitted model
    dataframe["expected_average_profit"] = ggf.conditional_expected_average_profit(dataframe['frequency'],
                                                                                   dataframe['monetary'])

    # 4. Calculate CLTV with BG-NBD and GG model.
    # Initialize Beta-Geometric/Negative Binomial Distribution Fitter and fit it to the data
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(dataframe['frequency'], dataframe['recency'], dataframe['T'])

    # Calculate customer lifetime value using the fitted models
    cltv = ggf.customer_lifetime_value(bgf,
                                       dataframe['frequency'],
                                       dataframe['recency'],
                                       dataframe['T'],
                                       dataframe['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    # Convert the resulting CLTV into a pandas dataframe, merge it with the original dataframe,
    # and add a segment column based on quartiles of the CLTV
    cltv = cltv.reset_index()
    cltv_final = dataframe.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    # Return the final dataframe with CLTV and segment columns added
    return cltv_final

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")
