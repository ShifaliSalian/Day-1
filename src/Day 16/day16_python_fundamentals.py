# ==========================================================
# DAY 16 — STATISTICS: DISTRIBUTIONS COMPLETE SCRIPT
# ==========================================================

# ----------------------------------------------------------
# STEP 1 — Import Required Libraries
# ---------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ----------------------------------------------------------
# STEP 2 — Create Dataset (Simulated Real-World Data)
# ----------------------------------------------------------
# We'll generate both normal and skewed data to compare

np.random.seed(42)

normal_data = np.random.normal(loc=50, scale=10, size=1000)   # Normal distribution
skewed_data = np.random.exponential(scale=20, size=1000)      # Right-skewed distribution

df = pd.DataFrame({
    "Normal_Data": normal_data,
    "Skewed_Data": skewed_data
})

# ==========================================================
# TOPIC 1 — UNDERSTANDING DISTRIBUTIONS
# ==========================================================

# Histogram for Normal Distribution
plt.figure()
sns.histplot(df["Normal_Data"], kde=True)
plt.title("Normal Distribution")
plt.show()

# Histogram for Skewed Distribution
plt.figure()
sns.histplot(df["Skewed_Data"], kde=True)
plt.title("Right-Skewed Distribution")
plt.show()

# Compare Mean and Median
print("\nNormal Data Mean:", df["Normal_Data"].mean())
print("Normal Data Median:", df["Normal_Data"].median())

print("\nSkewed Data Mean:", df["Skewed_Data"].mean())
print("Skewed Data Median:", df["Skewed_Data"].median())

# ==========================================================
# TOPIC 2 — Z-SCORES & OUTLIER DETECTION
# ==========================================================

# Calculate Z-scores for Normal Data
mean = df["Normal_Data"].mean()
std = df["Normal_Data"].std()

df["Z_Score"] = (df["Normal_Data"] - mean) / std

# Identify potential outliers (|Z| > 3)
outliers = df[np.abs(df["Z_Score"]) > 3]

print("\nNumber of Outliers Detected:", len(outliers))

# Visualize Outliers using Boxplot
plt.figure()
sns.boxplot(x=df["Normal_Data"])
plt.title("Outlier Detection (Boxplot)")
plt.show()

# ==========================================================
# TOPIC 3 — CENTRAL LIMIT THEOREM (CLT)
# ==========================================================

sample_means = []

# Take many random samples and compute their means
for i in range(1000):
    sample = np.random.choice(df["Skewed_Data"], size=30)
    sample_means.append(sample.mean())

# Plot distribution of sample means
plt.figure()
sns.histplot(sample_means, kde=True)
plt.title("Central Limit Theorem — Distribution of Sample Means")
plt.show()

# ==========================================================
# TOPIC 4 — SAMPLING TECHNIQUES
# ==========================================================

# Treat Normal_Data as population
population = df["Normal_Data"]

# Take random sample
sample = population.sample(n=50, random_state=42)

# Compare statistics
print("\nPopulation Mean:", population.mean())
print("Sample Mean:", sample.mean())

print("\nPopulation Std:", population.std())
print("Sample Std:", sample.std())

# Visualize population vs sample
plt.figure()
sns.histplot(population, color="blue", label="Population", kde=True)
sns.histplot(sample, color="red", label="Sample", kde=True)
plt.legend()
plt.title("Population vs Sample Distribution")
plt.show()

# ==========================================================
# FINAL INSIGHTS (Students should write their own)
# ==========================================================
print("\n===== SAMPLE INSIGHTS =====")
print("1. Normal data is symmetric, skewed data has long right tail.")
print("2. Z-scores help identify unusual values.")
print("3. Sample means become normally distributed (CLT).")
print("4. Sample statistics approximate population statistics.")
print("\nDay 16 Completed Successfully!")
"""

"""
# Task 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
heights = np.random.normal(loc=170, scale=10, size=1000)               
incomes = np.random.lognormal(mean=10, sigma=0.5, size=1000)           
test_scores = 100 - np.random.beta(a=2, b=8, size=1000) * 100          
datasets = {
    "Human Heights (Normal)": heights,
    "Household Incomes (Right-Skewed)": incomes,
    "Test Scores (Left-Skewed)": test_scores
}
plt.figure(figsize=(15, 4))
for i, (name, values) in enumerate(datasets.items(), 1):
    s = pd.Series(values)
    mean_val = s.mean()
    median_val = s.median()
    plt.subplot(1, 3, i)
    plt.hist(s, bins=30, density=True, alpha=0.6)
    s.plot(kind="kde")
    plt.title(f"{name}\nMean={mean_val:.2f}, Median={median_val:.2f}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
plt.tight_layout()
plt.show()
for name, values in datasets.items():
    s = pd.Series(values)
    mean_val = s.mean()
    median_val = s.median()
    if mean_val > median_val:
        print(f"{name}: Right-Skewed (Mean > Median)")
    elif mean_val < median_val:
        print(f"{name}: Left-Skewed (Mean < Median)")
    else:
        print(f"{name}: Approximately Normal (Mean ≈ Median)")



# Task 2
import pandas as pd
import numpy as np

np.random.seed(42)

scores = np.random.normal(loc=70, scale=10, size=1000)

scores = np.append(scores, [20, 120])

data = pd.DataFrame({'Score': scores})

mu = data['Score'].mean()
sigma = data['Score'].std()

print(f"Mean (μ): {mu:.2f}")
print(f"Standard Deviation (σ): {sigma:.2f}")

data['z_score'] = (data['Score'] - mu) / sigma

outliers = data[np.abs(data['z_score']) > 3]

print("\nStatistical Outliers (|Z| > 3):")
print(outliers)


# Task 3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Create Skewed Dataset
# -----------------------------
# Exponential distribution (right-skewed)
original_data = np.random.exponential(scale=2, size=10000)

# Plot original distribution
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(original_data, kde=True)
plt.title("Original Skewed Distribution")

# -----------------------------
# Step 2: Sampling Process
# -----------------------------
sample_means = []

for _ in range(1000):
    sample = np.random.choice(original_data, size=30)
    sample_means.append(np.mean(sample))

# -----------------------------
# Step 3: Plot Distribution of Means
# -----------------------------
plt.subplot(1,2,2)
sns.histplot(sample_means, kde=True)
plt.title("Distribution of Sample Means (CLT)")

plt.tight_layout()
plt.show()
















