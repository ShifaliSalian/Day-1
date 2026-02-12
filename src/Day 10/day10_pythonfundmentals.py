#Task 1
import pandas as pd
df=pd.read_csv("customer_order.csv")
print("First rows:\n", df.head())
print("Shape of the dataset:", df.shape)
print("\nMissing values per column:")
print(df.isna().sum())
df["order_id"] = df["order_id"].fillna(df["order_id"].median())
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())
df = df.drop_duplicates()
print("\nShape after removing duplicates:", df.shape)

# Task 2
import pandas as pd
df = pd.read_csv("sample_price_data.csv")
print("Before cleaning:")
print(df.dtypes)
df["Price"] = df["Price"].str.replace("$", "", regex=False).astype(float)
df["Date"] = pd.to_datetime(df["Date"])
print("\nAfter cleaning:")
print(df.dtypes)
print("\nAverage Price:", df["Price"].mean())

# Task 3
import pandas as pd
df = pd.read_csv("location_dirty_data.csv")
print(df["Location"].unique())
df["Location"] = df["Location"].str.strip()
df["Location"] = df["Location"].str.title() 
print("Unique location values:")
print(df["Location"].unique())
