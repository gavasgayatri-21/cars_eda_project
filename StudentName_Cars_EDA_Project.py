"""
Exploratory Data Analysis on a Cars Dataset
===========================================
Dataset : CarDekho Used Car dataset (India), 15,411 listings
Source  : https://github.com/manishkr1754/CarDekho_Used_Car_Price_Prediction
          (file: notebooks/data/cardekho_dataset.csv, originally scraped from cardekho.com)

How to run
----------
    pip install pandas numpy matplotlib seaborn
    python StudentName_Cars_EDA_Project.py

The script looks for 'cardekho_dataset.csv' next to this file. If it is not
found, it downloads the file from the URL above. All charts and summary tables
are written to the folder 'eda_outputs/'.
"""

# =============================================================================
# 0. SETUP
# =============================================================================
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")                       # save charts to files (no pop-up windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.0)

STUDENT_NAME = "Your Name"                  # <- change this
DATA_FILE = "cardekho_dataset.csv"
DATA_URL = ("https://raw.githubusercontent.com/manishkr1754/"
            "CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv")
OUT_DIR = "eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

LAKH = 100_000                              # 1 lakh = 100,000 INR


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def save_chart(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 1. LOAD DATA & FIRST INSPECTION
# =============================================================================
section("1. DATA LOADING AND INSPECTION")

path = DATA_FILE if os.path.exists(DATA_FILE) else DATA_URL
raw = pd.read_csv(path)

print("Shape (rows, columns):", raw.shape)
print("\nColumn names and data types:")
print(raw.dtypes)
print("\nSample rows:")
print(raw.head())

# =============================================================================
# 2. DATA QUALITY CHECKS (before cleaning)
# =============================================================================
section("2. DATA QUALITY CHECKS")

df = raw.copy()
# 'Unnamed: 0' is just the original row index saved to CSV - it carries no information
df = df.drop(columns=["Unnamed: 0"])

missing = df.isna().sum()
print("Missing values per column:\n", missing[missing > 0] if missing.any() else "None found")

dup_count = int(df.duplicated().sum())
print("\nExact duplicate rows (ignoring the index column):", dup_count)

print("\nBrand labels (check for inconsistent spelling):")
print(sorted(df["brand"].unique()))
print("\nSeller types :", df["seller_type"].unique().tolist())
print("Fuel types   :", df["fuel_type"].unique().tolist())
print("Transmission :", df["transmission_type"].unique().tolist())
print("Seats values :", sorted(df["seats"].unique().tolist()))

# Suspicious records worth a closer look
print("\nRecords with 0 seats:", int((df["seats"] == 0).sum()))
print("Records with km_driven > 1,000,000:", int((df["km_driven"] > 1_000_000).sum()))
print(df.loc[df["km_driven"] > 1_000_000, ["car_name", "vehicle_age", "km_driven", "selling_price"]])
print("\nElectric cars mileage values (kmpl is not meaningful for EVs):")
print(df.loc[df["fuel_type"] == "Electric", ["car_name", "mileage"]])

# =============================================================================
# 3. DATA CLEANING (every decision is logged)
# =============================================================================
section("3. DATA CLEANING")

cleaning_log = [("Original dataset (incl. index column)", raw.shape[0], raw.shape[1])]
cleaning_log.append(("Dropped redundant index column 'Unnamed: 0'", len(df), df.shape[1]))

# 3.1 Standardise text labels: 'ISUZU' and 'Isuzu' are the same brand
for col in ["car_name", "brand", "model", "seller_type", "fuel_type", "transmission_type"]:
    df[col] = df[col].str.strip()
df["brand"] = df["brand"].replace({"ISUZU": "Isuzu"})
cleaning_log.append(("Standardised brand label 'ISUZU' -> 'Isuzu'", len(df), df.shape[1]))

# 3.2 Remove exact duplicate listings (same car, same specs, same price)
df = df.drop_duplicates().reset_index(drop=True)
cleaning_log.append(("Removed exact duplicate rows", len(df), df.shape[1]))

# 3.3 Remove impossible values: a car cannot have 0 seats
df = df[df["seats"] > 0]
cleaning_log.append(("Removed rows with 0 seats (invalid)", len(df), df.shape[1]))

# 3.4 Remove extreme km_driven entries (> 1,000,000 km) - almost certainly typing errors
df = df[df["km_driven"] <= 1_000_000].reset_index(drop=True)
cleaning_log.append(("Removed km_driven > 1,000,000 (likely entry errors)", len(df), df.shape[1]))

# 3.5 Mileage (kmpl) is not meaningful for electric cars -> mark as missing, do not impute
df.loc[df["fuel_type"] == "Electric", "mileage"] = np.nan
cleaning_log.append(("Set mileage of Electric cars to NaN (4 rows kept)", len(df), df.shape[1]))

# 3.6 Fix data types: text columns with few categories -> 'category'
for col in ["brand", "seller_type", "fuel_type", "transmission_type"]:
    df[col] = df[col].astype("category")

# 3.7 Feature engineering: price in lakh (easier to read) and km per year
df["price_lakh"] = df["selling_price"] / LAKH
df["km_per_year"] = df["km_driven"] / df["vehicle_age"].clip(lower=1)
cleaning_log.append(("Added engineered columns price_lakh, km_per_year", len(df), df.shape[1]))

# NOTE: Expensive luxury cars (Ferrari, Rolls-Royce, Bentley ...) are KEPT.
# They are genuine listings, not errors, and removing them would hide the real
# price range of the market. They are handled with log scales / medians instead.

print(f"{'Step':<58}{'Rows':>8}{'Cols':>6}")
for step, rows, cols in cleaning_log:
    print(f"{step:<58}{rows:>8}{cols:>6}")
print(f"\nSize BEFORE cleaning: {raw.shape}  |  AFTER cleaning: {df.shape}")

pd.DataFrame(cleaning_log, columns=["step", "rows", "columns"]).to_csv(
    os.path.join(OUT_DIR, "cleaning_log.csv"), index=False)

# =============================================================================
# 4. VARIABLE TYPES & DESCRIPTIVE STATISTICS
# =============================================================================
section("4. DESCRIPTIVE STATISTICS")

numeric_cols = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats", "selling_price"]
categorical_cols = ["brand", "seller_type", "fuel_type", "transmission_type"]
print("Numerical  :", numeric_cols)
print("Categorical:", categorical_cols + ["car_name", "model"])

num_desc = df[numeric_cols].describe().T
num_desc["skew"] = df[numeric_cols].skew()
num_desc = num_desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew"]].round(2)
print("\nNumerical summary:\n", num_desc)
num_desc.to_csv(os.path.join(OUT_DIR, "numeric_summary.csv"))

cat_desc = pd.DataFrame({
    "unique": [df[c].nunique() for c in categorical_cols],
    "most_common": [df[c].mode()[0] for c in categorical_cols],
    "top_share_%": [round(df[c].value_counts(normalize=True).iloc[0] * 100, 1) for c in categorical_cols],
}, index=categorical_cols)
print("\nCategorical summary:\n", cat_desc)
cat_desc.to_csv(os.path.join(OUT_DIR, "categorical_summary.csv"))

for c in ["seller_type", "fuel_type", "transmission_type"]:
    print(f"\n{c} share (%):\n", (df[c].value_counts(normalize=True) * 100).round(1))

# =============================================================================
# 5. VISUAL EDA  (8 charts, 5 chart types)
# =============================================================================
section("5. VISUAL EDA - saving charts to '" + OUT_DIR + "/'")

# ---- Chart 1: Bar plot - top 10 brands by number of listings (univariate) ----
top_brands = df["brand"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=top_brands.values, y=top_brands.index.astype(str), color="#2b6cb0", ax=ax)
for i, v in enumerate(top_brands.values):
    ax.text(v + 40, i, f"{v:,}", va="center", fontsize=9)
ax.set_title("Chart 1: Top 10 Brands by Number of Listings")
ax.set_xlabel("Number of listings")
ax.set_ylabel("Brand")
save_chart(fig, "chart1_top_brands.png")

# ---- Chart 2: Bar plot - fuel type distribution (univariate) ----
fuel_counts = df["fuel_type"].value_counts()
fig, ax = plt.subplots(figsize=(8, 3.6))
sns.barplot(x=fuel_counts.index.astype(str), y=fuel_counts.values, color="#38a169", ax=ax)
for i, v in enumerate(fuel_counts.values):
    ax.text(i, v + 150, f"{v:,}\n({v / len(df) * 100:.1f}%)", ha="center", fontsize=9)
ax.set_title("Chart 2: Listings by Fuel Type")
ax.set_xlabel("Fuel type")
ax.set_ylabel("Number of listings")
ax.set_ylim(0, fuel_counts.max() * 1.18)
save_chart(fig, "chart2_fuel_type.png")

# ---- Chart 3: Histogram - price distribution, raw vs log scale (univariate) ----
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
p99 = df["price_lakh"].quantile(0.99)
sns.histplot(df.loc[df["price_lakh"] <= p99, "price_lakh"], bins=50, color="#dd6b20", ax=axes[0])
axes[0].axvline(df["price_lakh"].median(), color="black", ls="--", label=f"Median = {df['price_lakh'].median():.1f}")
axes[0].axvline(df["price_lakh"].mean(), color="red", ls="--", label=f"Mean = {df['price_lakh'].mean():.1f}")
axes[0].set_title("Price (up to 99th percentile)")
axes[0].set_xlabel("Selling price (INR lakh)")
axes[0].set_ylabel("Number of cars")
axes[0].legend()
sns.histplot(np.log10(df["selling_price"]), bins=50, color="#805ad5", ax=axes[1])
axes[1].set_title("Price on log10 scale (all cars)")
axes[1].set_xlabel("log10(Selling price in INR)")
axes[1].set_ylabel("Number of cars")
fig.suptitle("Chart 3: Distribution of Selling Price", fontsize=13, y=1.02)
save_chart(fig, "chart3_price_hist.png")

# ---- Chart 4: Box plot - price by transmission (bivariate) ----
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="transmission_type", y="price_lakh", palette=["#63b3ed", "#f6ad55"], ax=ax)
ax.set_yscale("log")
ax.set_title("Chart 4: Selling Price by Transmission Type (log scale)")
ax.set_xlabel("Transmission type")
ax.set_ylabel("Selling price (INR lakh, log scale)")
save_chart(fig, "chart4_price_transmission_box.png")

# ---- Chart 5: Scatter plots - vehicle age & km driven vs price (bivariate) ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.scatterplot(data=df, x="vehicle_age", y="price_lakh", alpha=0.25, s=14, color="#c53030", ax=axes[0])
age_med = df.groupby("vehicle_age")["price_lakh"].median()
age_med = age_med[df["vehicle_age"].value_counts().reindex(age_med.index) >= 20]
axes[0].plot(age_med.index, age_med.values, color="black", lw=2, label="Median price per age")
axes[0].set_yscale("log")
axes[0].set_title("Vehicle age vs price")
axes[0].set_xlabel("Vehicle age (years)")
axes[0].set_ylabel("Selling price (INR lakh, log scale)")
axes[0].legend()
sns.scatterplot(data=df, x="km_driven", y="price_lakh", alpha=0.25, s=14, color="#2c7a7b", ax=axes[1])
axes[1].set_yscale("log")
axes[1].set_title("Kilometres driven vs price")
axes[1].set_xlabel("Kilometres driven")
axes[1].set_ylabel("Selling price (INR lakh, log scale)")
axes[1].ticklabel_format(style="plain", axis="x")
fig.suptitle("Chart 5: Does Age or Distance Driven Explain Price?", fontsize=13, y=1.02)
save_chart(fig, "chart5_age_km_vs_price.png")

# ---- Chart 6: Bar plot - average price of brands with >= 30 listings (bivariate) ----
brand_stats = (df.groupby("brand", observed=True)["price_lakh"]
               .agg(["count", "mean", "median"]).query("count >= 30")
               .sort_values("mean", ascending=False))
top_price = brand_stats.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(top_price.index.astype(str), top_price["mean"], color="#d69e2e")
for i, v in enumerate(top_price["mean"]):
    ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)
ax.set_title("Chart 6: Top 10 Brands by Average Selling Price (brands with 30+ listings)")
ax.set_xlabel("Average selling price (INR lakh)")
ax.set_ylabel("Brand")
save_chart(fig, "chart6_brand_avg_price.png")
brand_stats.round(2).to_csv(os.path.join(OUT_DIR, "brand_price_stats.csv"))

# ---- Chart 7: Heatmap - correlation matrix (multivariate) ----
corr = df[numeric_cols].corr(method="pearson")
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"label": "Pearson correlation"}, ax=ax)
ax.set_title("Chart 7: Correlation Heatmap of Numerical Features")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
save_chart(fig, "chart7_correlation_heatmap.png")

# ---- Chart 8: Scatter - max power vs price coloured by fuel & transmission (multivariate) ----
main_fuels = df[df["fuel_type"].isin(["Petrol", "Diesel"])].copy()
for c in ["fuel_type", "transmission_type"]:
    main_fuels[c] = main_fuels[c].astype(str)      # drop unused categories for plotting
fig, ax = plt.subplots(figsize=(8.5, 4.6))
sns.scatterplot(data=main_fuels, x="max_power", y="price_lakh", hue="fuel_type",
                style="transmission_type", alpha=0.45, s=20, ax=ax,
                palette={"Petrol": "#e53e3e", "Diesel": "#2b6cb0"})
ax.set_yscale("log")
ax.set_title("Chart 8: Max Power vs Price by Fuel Type and Transmission")
ax.set_xlabel("Maximum power (bhp)")
ax.set_ylabel("Selling price (INR lakh, log scale)")
ax.legend(title="", fontsize=8, loc="lower right")
save_chart(fig, "chart8_power_price_multivariate.png")

print("8 charts saved.")

# =============================================================================
# 6. OUTLIERS (IQR rule) AND RELATIONSHIP STATISTICS
# =============================================================================
section("6. OUTLIERS AND RELATIONSHIPS")


def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()), q3 + 1.5 * iqr


outlier_table = {}
for col in ["selling_price", "km_driven", "mileage", "engine", "max_power", "vehicle_age"]:
    n_out, upper = iqr_outliers(df[col].dropna())
    outlier_table[col] = {"outliers": n_out, "pct": round(n_out / df[col].notna().sum() * 100, 1),
                          "upper_fence": round(float(upper), 1)}
print(pd.DataFrame(outlier_table).T)

print("\nTop 5 most expensive listings:")
print(df.nlargest(5, "selling_price")[["car_name", "vehicle_age", "km_driven", "selling_price"]])

spearman_age = df["vehicle_age"].corr(df["selling_price"], method="spearman")
spearman_km = df["km_driven"].corr(df["selling_price"], method="spearman")
print(f"\nSpearman (rank) correlation with price -> age: {spearman_age:.2f}, km_driven: {spearman_km:.2f}")

trans_median = df.groupby("transmission_type", observed=True)["price_lakh"].median()
trans_mean = df.groupby("transmission_type", observed=True)["price_lakh"].mean()
print("\nMedian price by transmission (lakh):\n", trans_median.round(2))

# =============================================================================
# 7. KEY FINDINGS (numbers are computed from the data, not typed by hand)
# =============================================================================
section("7. KEY FINDINGS")

n = len(df)
top2_share = df["brand"].isin(["Maruti", "Hyundai"]).mean() * 100
pd_share = df["fuel_type"].isin(["Petrol", "Diesel"]).mean() * 100
median_p, mean_p = df["price_lakh"].median(), df["price_lakh"].mean()
auto_ratio = trans_median["Automatic"] / trans_median["Manual"]
corr_price = corr["selling_price"].drop("selling_price").sort_values(key=abs, ascending=False)
mileage_by_fuel = df.groupby("fuel_type", observed=True)["mileage"].mean()

findings = [
    f"Market is dominated by a few brands: Maruti and Hyundai make up {top2_share:.1f}% of {n:,} listings.",
    f"Petrol and diesel cars account for {pd_share:.1f}% of listings; CNG, LPG and Electric are rare.",
    f"Prices are strongly right-skewed: median = {median_p:.2f} lakh vs mean = {mean_p:.2f} lakh "
    f"(skewness {df['selling_price'].skew():.1f}), pulled up by luxury cars.",
    f"Automatic cars have a median price {auto_ratio:.1f}x that of manual cars "
    f"({trans_median['Automatic']:.2f} vs {trans_median['Manual']:.2f} lakh).",
    f"max_power is the feature most strongly correlated with price (r = {corr_price['max_power']:.2f}), "
    f"followed by engine size (r = {corr_price['engine']:.2f}).",
    f"Vehicle age has a modest negative relationship with price (Pearson r = {corr['vehicle_age']['selling_price']:.2f}, "
    f"Spearman = {spearman_age:.2f}); km_driven is weaker still (Spearman = {spearman_km:.2f}).",
    f"Luxury brands lead on average price: {', '.join(brand_stats.head(3).index.astype(str))} "
    f"(averages of {', '.join(f'{v:.1f}' for v in brand_stats['mean'].head(3))} lakh).",
    f"Mileage correlates negatively with price (r = {corr['mileage']['selling_price']:.2f}) likely partly because powerful, "
    f"expensive cars burn more fuel (mileage vs max_power r = {corr['mileage']['max_power']:.2f}) - this is association, not cause.",
]
for i, f in enumerate(findings, 1):
    print(f"{i}. {f}")

print("\nLimitations:")
limitations = [
    "Asking prices from one platform (CarDekho), not final transaction prices.",
    "No city, condition, accident history or trim-level information, so price drivers are only partly captured.",
    "Only 4 electric and 44 LPG cars - no reliable conclusions can be drawn about them.",
    "Correlation does not prove causation; features such as engine size and power are themselves related.",
    "Dataset has no listing date; vehicle_age is relative to the scrape date.",
]
for item in limitations:
    print(" -", item)

# =============================================================================
# 8. SAVE SUMMARY FOR THE REPORT
# =============================================================================
summary = {
    "student": STUDENT_NAME,
    "rows_raw": int(raw.shape[0]), "cols_raw": int(raw.shape[1]),
    "rows_clean": int(df.shape[0]), "cols_clean": int(df.shape[1]),
    "duplicates_removed": dup_count,
    "missing_total": int(missing.sum()),
    "findings": findings, "limitations": limitations,
    "outliers": outlier_table,
    "corr_with_price": corr_price.round(3).to_dict(),
    "spearman_age": round(float(spearman_age), 3), "spearman_km": round(float(spearman_km), 3),
    "trans_median": trans_median.round(2).to_dict(), "trans_mean": trans_mean.round(2).to_dict(),
    "fuel_share": (df["fuel_type"].value_counts(normalize=True) * 100).round(1).to_dict(),
    "brand_share_top5": (df["brand"].value_counts(normalize=True).head(5) * 100).round(1).to_dict(),
    "age_median_price": df.groupby("vehicle_age")["price_lakh"].median().loc[[1, 3, 5, 8, 10, 15]].round(2).to_dict(),
    "mileage_by_fuel": mileage_by_fuel.round(2).to_dict(),
    "brand_top_price": brand_stats.head(5)[["count", "mean", "median"]].round(2).reset_index().to_dict("records"),
    "median_price": round(float(median_p), 2), "mean_price": round(float(mean_p), 2),
    "p99_price": round(float(p99), 2), "max_price": round(float(df["price_lakh"].max()), 2),
    "min_price": round(float(df["price_lakh"].min()), 2),
    "km_median": float(df["km_driven"].median()),
    "power_price_corr_petrol_diesel": round(float(main_fuels["max_power"].corr(main_fuels["selling_price"])), 3),
    "diesel_vs_petrol_median": df.groupby("fuel_type", observed=True)["price_lakh"].median().round(2).to_dict(),
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=str)

df.to_csv(os.path.join(OUT_DIR, "cars_cleaned.csv"), index=False)
print(f"\nDone. Charts and tables saved in '{OUT_DIR}/'.")
