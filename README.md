# Exploratory Data Analysis on a Cars Dataset

A complete EDA of **15,411 used-car listings from CarDekho (India)**: data inspection, cleaning, descriptive statistics, eight charts, and eight evidence-based findings. No machine-learning model is used.

> Student project for Unlox Academy - *Exploratory Data Analysis on a Cars Dataset*.

---

## Dataset

| Item | Detail |
|---|---|
| Name | CarDekho Used Car dataset |
| Source | CarDekho used-car listings (cardekho.com), obtained from the public GitHub repository [`manishkr1754/CarDekho_Used_Car_Price_Prediction`](https://github.com/manishkr1754/CarDekho_Used_Car_Price_Prediction), file `notebooks/data/cardekho_dataset.csv` |
| Size | 15,411 rows x 14 columns (13 data columns + 1 saved index column) |
| Key columns | `brand`, `model`, `vehicle_age`, `km_driven`, `seller_type`, `fuel_type`, `transmission_type`, `mileage`, `engine`, `max_power`, `seats`, `selling_price` |

The dataset is **not included** in the submission. The script reads `cardekho_dataset.csv` if it sits next to the script; otherwise it downloads the file from the URL above automatically (internet connection needed on first run).

---

## Files

| File | Purpose |
|---|---|
| `StudentName_Cars_EDA_Project.py` | The full analysis: loading, cleaning, statistics, charts, findings |
| `Cars_EDA_Project_Report.pdf` | 8-page project report with charts, observations, findings and conclusion |
| `README.md` | This file |

---

## How to run

```bash
pip install pandas numpy matplotlib seaborn scipy
python StudentName_Cars_EDA_Project.py
```

Tested with Python 3.12. Running the script prints each step to the console and creates a folder **`eda_outputs/`** containing:

- `chart1_...png` to `chart8_...png` - the eight charts
- `cleaning_log.csv`, `numeric_summary.csv`, `categorical_summary.csv`, `brand_price_stats.csv` - summary tables
- `cars_cleaned.csv` - the cleaned dataset
- `summary.json` - key numbers used in the report

---

## Workflow

The script is divided into numbered sections:

1. **Data loading and inspection** - shape, columns, data types, sample rows
2. **Data quality checks** - missing values, duplicates, inconsistent labels, suspicious values
3. **Data cleaning** - each decision is logged with the row count after the step
4. **Descriptive statistics** - numerical and categorical summaries
5. **Visual EDA** - eight charts
6. **Outliers and relationships** - IQR rule, Spearman correlations
7. **Key findings** - computed from the data (not typed by hand)
8. **Save summary** for the report

### Cleaning summary

| Step | Rows left |
|---|---|
| Original data | 15,411 |
| Removed 167 exact duplicates | 15,244 |
| Removed 2 cars with `seats = 0` (invalid) | 15,242 |
| Removed 2 cars with `km_driven` > 1,000,000 (likely entry errors) | 15,240 |

Other decisions: dropped the redundant index column; merged brand labels `ISUZU` / `Isuzu`; set `mileage` of 4 electric cars to missing (kmpl is meaningless for EVs); **kept** luxury-car price outliers because they are genuine listings.

### Charts

| # | Chart | Type | Analysis |
|---|---|---|---|
| 1 | Top 10 brands by listings | Bar | Univariate |
| 2 | Listings by fuel type | Bar | Univariate |
| 3 | Price distribution (raw and log) | Histogram | Univariate |
| 4 | Price by transmission | Box plot | Bivariate |
| 5 | Age and km driven vs price | Scatter | Bivariate |
| 6 | Average price by brand | Bar | Bivariate |
| 7 | Correlation matrix | Heatmap | Multivariate |
| 8 | Max power vs price by fuel and transmission | Scatter | Multivariate |

---

## Key findings

1. Maruti and Hyundai make up **51.7%** of listings; the market is concentrated in mass-market brands.
2. Petrol and diesel account for **97.7%** of listings.
3. Price is strongly right-skewed: **median 5.59 lakh vs mean 7.75 lakh**.
4. Automatic cars have a median price about **2.1x** that of manual cars (10.5 vs 5.0 lakh), partly because automatics are concentrated in premium models.
5. `max_power` is the feature most correlated with price (**r = 0.75**).
6. Age matters more than distance driven: median price falls from about 7.5 lakh (1-3 years) to 1.4 lakh (15 years), while `km_driven` has only a weak relationship with price.
7. Land Rover, BMW, Jaguar and Mercedes-Benz have the highest average prices.
8. Mileage is negatively related to price, likely because powerful cars are less fuel-efficient (association, not cause).

## Limitations

- Prices are asking prices from one platform, not final sale prices.
- No city, condition, accident history, trim or listing-date information.
- Only 4 electric and 44 LPG cars - no reliable conclusions for these.
- Correlation does not imply causation.

---

## Before submitting

- Rename the code file to `YourName_Cars_EDA_Project.py` and set `STUDENT_NAME` at the top of the script.
- Write your name and batch on the report cover page.
- Make sure you can explain every step of the workflow.

*1 lakh = INR 100,000.*
