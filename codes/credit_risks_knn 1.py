import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#df = pd.read_csv("datasets/dataset_31_credit-g.arff")
data = fetch_openml(data_id=31, as_frame=True)
df = data.frame
print(df.head())
#print(df.columns)
#print(df.info())
df.dropna(inplace=True)
numeric_features = ["duration", "credit_amount", "age", "installment_commitment"]
categorical_features = ["checking_status", "credit_history", "purpose", "savings_status"]
target = "class"
X = df[numeric_features + categorical_features]
y = df[target]
print(df[target])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
preprocessor = ColumnTransformer([("num", StandardScaler(), numeric_features), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)])
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
test_accuracies = {}
print("Test set accuracies:")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_proc, y_train)
    preds = knn.predict(X_test_proc)
    acc = accuracy_score(y_test, preds)
    test_accuracies[k] = acc
    print(f"  k = {k:2d} → accuracy = {acc:.4f}")
best_k = max(test_accuracies, key=test_accuracies.get)
print(f"\nBest k: {best_k} (accuracy = {test_accuracies[best_k]:.4f})")
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_proc, y_train)
y_pred = best_knn.predict(X_test_proc)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (rows=true class, cols=predicted class):")
print(cm)