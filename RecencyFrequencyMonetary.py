# Customer Segmentation with RFM

# İmport necessary libraries
import datetime as dt
import pandas as pd

# Set options for display of pandas dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Step 1.
# Business Problem and Data Understanding

# Load the online retail data
df = pd.read_excel(r"CRM_Analytics\online_retail_II.xlsx", sheet_name="Year 2009-2010")

# Display the first 5 rows of the data
df.head()

# Print the shape and number of missing values of the data
print("Data shape: ", df.shape)
print("Missing values: ", df.isnull().sum(), sep='\n')

# Count the number of unique items
print("Unique items count: ", df["Description"].nunique())

# Count the number of each item
print("Item count: ", df["Description"].value_counts(), sep='\n')

# Sum the quantity of each item
print("Sum of item quantity: ", df.groupby("Description").agg({"Quantity": "sum"}))

# Find the items with the highest total quantity
print("Top items by total quantity: ", df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head())

# Count the number of unique invoices
print("Unique invoices count: ", df["Invoice"].nunique())

# Step 2.
# Data Freparation

# Clean the data by removing negative quantities, missing values and canceled invoices
df = df[(df['Quantity'] > 0)]

# Calculate the total price for each invoice
df["TotalPrice"] = df["Quantity"] * df["Price"]
print("Total price for each invoice: ", df.groupby("Invoice").agg({"TotalPrice": "sum"}).head())

# Filter out rows where the "Invoice" column does not contain "C"
df = df[~df["Invoice"].str.contains("C", na=False)]

# Remove any rows with missing or NaN values
df.dropna(inplace=True)

# Step 3.
# Calculating RFM Metrics

# Get the maximum Invoice date from the data
max_invoice_date = df["InvoiceDate"].max()

# Set the analysis date
analysis_date = max_invoice_date + + dt.timedelta(days=2)

# Group the data by the customer ID and calculate the recency, frequency, and monetary values
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (analysis_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# Rename the columns for readability
rfm.columns = ['recency', 'frequency', 'monetary']

# Filter out customers who have a monetary value of 0
rfm = rfm[rfm["monetary"] > 0]

# Step 4.
# Calculate RFM Scores

# Calculate the recency score
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# Calculate the frequency score
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# Calculate the monetary score
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combine the scores to get the RFM score
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

# Step 5.
# Create & Analyze RFM Segments

# Define a mapping for the RFM segments
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

# Add a new column to the dataframe to hold the customer segment names
rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

# Keep only the relevant columns in the dataframe
rfm = rfm[["recency", "frequency", "monetary", "segment"]]

# Group by 'segment' and calculate mean and count for each RFM metric
rfm.groupby("segment").agg(["mean", "count"])

# Create an empty DataFrame 
new_customers_df = pd.DataFrame()

# Select the index of "new_customers" segment and add to `new_customers_df`
new_customers_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

# Save the new customers to a csv file
new_customers_df.to_csv("new_customers.csv")


# Functionalize all steps
def create_rfm(dataframe, csv=False):
    """
    Create the RFM (recency, frequency, monetary) scores for a dataframe of customer data

    dataframe : pandas DataFrame : input dataframe
    csv : boolean : whether to output to csv

    return : pandas DataFrame : RFM scores
    """

    # Prepare data
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True, axis=0) # drop rows with missing values
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)] # keep rows that do not contain 'C' in Invoice

    # Calculate RFM metrics
    analysis_date = df["InvoiceDate"].max() + + dt.timedelta(days=2)
    rfm = dataframe.groupby('Customer ID').agg({
        'InvoiceDate': lambda date: (analysis_date - date.max()).days, # calculate difference between analysis_date and max InvoiceDate
        'Invoice': lambda num: num.nunique(), # number of unique Invoices
        "TotalPrice": lambda price: price.sum() # sum of TotalPrice
    })
    rfm.columns = ['recency', 'frequency', "monetary"] # rename columns
    rfm = rfm[(rfm['monetary'] > 0)] # keep rows where monetary is positive

    # Calculate RFM scores
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1]) # score recency in 5 categories
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]) # score frequency in 5 categories
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5]) # score monetary in 5 categories

    # Create RFM_SCORE by concatenating recency_score and frequency_score
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # Assign customer segments based on RFM_SCORE
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    # Save the result to csv file if csv=True
    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

rfm = create_rfm(df)