#EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

data = {
    "Age": [25,30,35,40,28,32,45,50,23,36,29,41],
    "Salary": [30000,40000,50000,65000,42000,48000,80000,90000,28000,52000,46000,70000],
    "Experience": [1,3,7,10,2,5,15,20,1,8,4,12],
    "Department": ["IT","HR","IT","Finance","HR","IT","Finance","Finance","HR","IT","HR","Finance"],
    "Gender": ["M","F","M","M","F","F","M","M","F","F","M","F"]
}

df=pd.DataFrame(data)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# Dataset shape
print("\nDataset shape (rows, columns):", df.shape)

# Data types and missing values
print("\nDataset info:")
print(df.info())

# Summary statistics for numerical columns
print("\nSummary statistics:")
print(df.describe())

# ----------------------------------------------------------
# TOPIC 2 — UNIVARIATE ANALYSIS
# Analyze ONE variable at a time
# ----------------------------------------------------------

# HISTOGRAM — Distribution of Age
plt.figure()
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

# HISTOGRAM — Distribution of Salary
plt.figure()
sns.histplot(df["Salary"], kde=True)
plt.title("Salary Distribution")
plt.show()

# BOXPLOT — Detect spread & outliers in Salary
plt.figure()
sns.boxplot(x=df["Salary"])
plt.title("Salary Boxplot")
plt.show()

# CATEGORICAL ANALYSIS — Frequency counts
print("\nDepartment counts:")
print(df["Department"].value_counts())

print("\nGender counts:")
print(df["Gender"].value_counts())

# Bar plot for categorical variable
plt.figure()
sns.countplot(x="Department", data=df)
plt.title("Department Distribution")
plt.show()

# ----------------------------------------------------------
# TOPIC 3 — BIVARIATE ANALYSIS
# Study relationship between TWO variables
# ----------------------------------------------------------

# SCATTER PLOT — Age vs Salary
plt.figure()
sns.scatterplot(x="Age", y="Salary", data=df)
plt.title("Age vs Salary")
plt.show()

# SCATTER PLOT — Experience vs Salary
plt.figure()
sns.scatterplot(x="Experience", y="Salary", data=df)
plt.title("Experience vs Salary")
plt.show()

# BOXPLOT — Salary by Gender
plt.figure()
sns.boxplot(x="Gender", y="Salary", data=df)
plt.title("Salary by Gender")
plt.show()

# BOXPLOT — Salary by Department
plt.figure()
sns.boxplot(x="Department", y="Salary", data=df)
plt.title("Salary by Department")
plt.show()

# ----------------------------------------------------------
# TOPIC 4 — CORRELATION ANALYSIS
# ----------------------------------------------------------

# Correlation matrix (numerical columns only)
corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix)

# HEATMAP — visualize correlations
plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------------------------------
# TOPIC 5 — OUTLIER DETECTION
# ----------------------------------------------------------

# Boxplot for Age
plt.figure()
sns.boxplot(x=df["Age"])
plt.title("Age Outliers")
plt.show()

# Boxplot for Experience
plt.figure()
sns.boxplot(x=df["Experience"])
plt.title("Experience Outliers")
plt.show()

# ----------------------------------------------------------
# FINAL STEP — DOCUMENT INSIGHTS (PRINT SAMPLE INSIGHTS)
# Students should write their own observations here.
# ----------------------------------------------------------

print("\n===== SAMPLE INSIGHTS =====")
print("1. Salary increases with Experience and Age (positive correlation).")
print("2. Finance department shows higher salary range.")
print("3. No extreme outliers detected in Age or Experience.")
print("4. Gender distribution appears balanced.")
print("5. Experience strongly influences Salary.")


# Task 1
import pandas as pd
import matplotlib.pyplot as plt
data = {
    "Price": [250000, 270000, 300000, 320000, 350000, 400000, 450000, 500000,600000, 750000, 900000, 1200000], 
    "City": ["Delhi", "Delhi", "Mumbai", "Mumbai", "Delhi", "Bangalore","Bangalore", "Mumbai", "Delhi", "Delhi", "Mumbai", "Delhi"]
}
df = pd.DataFrame(data)
plt.figure(figsize=(8, 5))
plt.hist(df["Price"], bins=6, density=True, alpha=0.6)
df["Price"].plot(kind="kde")
plt.title("Histogram + KDE of House Prices")
plt.xlabel("Price")
plt.ylabel("Density")
plt.grid(True)
plt.show()
skewness = df["Price"].skew()
kurtosis = df["Price"].kurt()
print("Skewness:", skewness)
print("Kurtosis:", kurtosis)
city_counts = df["City"].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(city_counts.index, city_counts.values)
plt.title("Count Plot of City")
plt.xlabel("City")
plt.ylabel("Count")
plt.grid(axis="y")
plt.show()

# Task 2
import pandas as pd
import matplotlib.pyplot as plt
data = {
    "SquareFootage": [600, 750, 900, 1100, 1300, 1500, 1700, 2000, 2300, 2600],
    "Price":         [35, 45, 55, 70, 85, 95, 110, 140, 160, 190],  
    "City": ["Delhi", "Delhi", "Mumbai", "Mumbai", "Delhi",
             "Bangalore", "Bangalore", "Mumbai", "Delhi", "Mumbai"]
}
df = pd.DataFrame(data)
plt.figure(figsize=(7, 5))
plt.scatter(df["SquareFootage"], df["Price"])
plt.title("Scatter Plot: SquareFootage vs Price")
plt.xlabel("Square Footage")
plt.ylabel("Price (in Lakhs)")
plt.grid(True)
plt.show()
plt.figure(figsize=(7, 5))
df.boxplot(column="Price", by="City")
plt.title("Boxplot: City vs Price")
plt.suptitle("")  
plt.xlabel("City")
plt.ylabel("Price (in Lakhs)")
plt.grid(True)
plt.show()
print("Observation: As SquareFootage increases, Price seems to increase.")


# Task 3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = {
    "SquareFootage": [600, 750, 900, 1100, 1300, 1500, 1700, 2000, 2300, 2600],
    "Bedrooms":      [1, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    "Bathrooms":     [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    "Price":         [35, 45, 55, 70, 85, 95, 110, 140, 160, 300]  # outlier at 300
}
df = pd.DataFrame(data)
corr_matrix = df.corr(numeric_only=True)
print("Correlation Matrix:\n")
print(corr_matrix)
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        value = corr_matrix.iloc[i, j]
        if value > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], value))

print("\nHighly correlated pairs (> 0.8):")
for pair in high_corr_pairs:
    print(pair)
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["Price"])
plt.title("Boxplot for Outliers in Price")
plt.ylabel("Price")
plt.show()






























