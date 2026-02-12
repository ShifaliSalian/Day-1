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
