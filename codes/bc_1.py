import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv("datasets/data.csv")

for col in ["Unnamed: 32", "unnamed: 32", "Unnamed:32"]:
    if col in df.columns:
        df = df.drop(columns=[col])

missing_before = df.isna().sum().sum()
df = df.dropna()
missing_after = df.isna().sum().sum()

print(f"Missing values before drop: {missing_before}")
print(f"Missing values after drop:  {missing_after}")

target_col = "diagnosis"
candidate_features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean"
]
drop_cols = [c for c in ["id"] if c in df.columns]
df = df.drop(columns=drop_cols)

kept = [c for c in candidate_features if c in df.columns]
cols = [target_col] + kept
df = df[cols].copy()

print(f"Using {len(kept)} features: {kept}")

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

X = df[kept].copy()
y = df[target_col].copy()

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)

df_refined = pd.DataFrame(X_scaled, columns=kept)
df_refined[target_col] = y.values

out_path = "data_refined.csv"
df_refined.to_csv(out_path, index=False)
print(f"Saved refined dataset to {out_path}")

pairplot_features = kept[:5]
df_pair = df_refined[pairplot_features + [target_col]].copy()
df_pair[target_col] = df_pair[target_col].map({0: "Benign", 1: "Malignant"})

sns.pairplot(df_pair, hue=target_col, diag_kind="hist", height=2.0)
plt.suptitle("Pair Plot (subset of features)", y=1.02)
plt.show()

corr = df_refined[kept].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=False, cmap="viridis")
plt.title("Correlation Matrix (Scaled Features)")
plt.tight_layout()
plt.show()

df_long = df_refined.melt(id_vars=[target_col], value_vars=kept, var_name="feature", value_name="value")
df_long[target_col] = df_long[target_col].map({0: "Benign", 1: "Malignant"})

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_long, x="feature", y="value", hue=target_col)
plt.title("Box Plots by Diagnosis")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
print("Done")