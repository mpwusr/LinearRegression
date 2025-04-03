import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('kc_house_data.csv')

all_columns = df.columns.tolist()
print("All columns in dataset:", all_columns)

features_all = [col for col in all_columns if col not in ['id', 'price']]
print("Number of features used for training:", len(features_all))  # Should be 19

num_examples = df.shape[0]
print("Number of examples in the dataset:", num_examples)  # Should be 21613

X_bedrooms = df[['bedrooms']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X_bedrooms, y, test_size=0.2, random_state=42)

model_bedrooms = LinearRegression()
model_bedrooms.fit(X_train, y_train)

r2_train_bedrooms = model_bedrooms.score(X_train, y_train)
r2_test_bedrooms = model_bedrooms.score(X_test, y_test)
print("\nSimple Linear Regression (bedrooms):")
print("R² training =", r2_train_bedrooms)
print("R² testing  =", r2_test_bedrooms)

feature_set = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront']
r2_results = {}
for feature in feature_set:
    X_feature = df[[feature]]
    X_tr, X_te, y_tr, y_te = train_test_split(X_feature, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    r2 = model.score(X_te, y_te)
    r2_results[feature] = r2
    print(f"Feature: {feature}, R² testing = {r2}")

best_feature = max(r2_results, key=r2_results.get)
print(f"\nBest single feature is '{best_feature}' with R² testing = {r2_results[best_feature]}")

X_best = df[[best_feature]]
X_tr, X_te, y_tr, y_te = train_test_split(X_best, y, test_size=0.2, random_state=42)
model_best = LinearRegression()
model_best.fit(X_tr, y_tr)

plt.figure(figsize=(8, 6))
plt.scatter(df[best_feature], y, alpha=0.3, label='Data points')

x_line = np.linspace(df[best_feature].min(), df[best_feature].max(), 100).reshape(-1, 1)
x_line_df = pd.DataFrame(x_line, columns=[best_feature])
y_line = model_best.predict(x_line_df)
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.xlabel(best_feature)
plt.ylabel("Price")
plt.title(f"Scatterplot of {best_feature} vs Price")
plt.legend()
plt.show()

features_multi = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_above',
                  'floors', 'waterfront', 'view', 'sqft_basement', 'lat']
X_multi = df[features_multi]
X_tr, X_te, y_tr, y_te = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_tr, y_tr)

r2_train_multi = model_multi.score(X_tr, y_tr)
r2_test_multi = model_multi.score(X_te, y_te)
print("\nMultiple Regression Model:")
print("R² training =", r2_train_multi)
print("R² testing  =", r2_test_multi)
