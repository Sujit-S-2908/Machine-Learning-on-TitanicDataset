import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

data = pd.read_csv("train.csv")

# Labeling Sex into binary values
label_encoder = {}
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
label_encoder['Sex'] = LabelEncoder()
print(data)

# Handle missing values by filling with mean
num_fea = data.select_dtypes(include=['int', 'float'])
data = data.fillna(num_fea.mean())

# Preparing data for classification (Logistic Regression, Naive Bayes, Decision Tree, Random Forest)
x_class = np.array(data['Sex']).reshape(-1, 1)
y_class = np.array(data['Survived'])
x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(x_class, y_class, test_size=0.20, random_state=42)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(x_train_class, y_train_class)
y_pred_class_log = log_model.predict(x_test_class)
acc_class_log = accuracy_score(y_test_class, y_pred_class_log)
print("Logistic Regression Accuracy:", acc_class_log * 100)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train_class, y_train_class)
y_pred_class_nb = nb_model.predict(x_test_class)
acc_class_nb = accuracy_score(y_test_class, y_pred_class_nb)
print("Naive Bayes Accuracy:", acc_class_nb * 100)

# Decision Tree
dt_model_class = DecisionTreeClassifier()
dt_model_class.fit(x_train_class, y_train_class)
y_pred_class_dt = dt_model_class.predict(x_test_class)
acc_class_dt = accuracy_score(y_test_class, y_pred_class_dt)
print("Decision Tree Classification Accuracy:", acc_class_dt * 100)

# Random Forest
rf_model_class = RandomForestClassifier()
rf_model_class.fit(x_train_class, y_train_class)
y_pred_class_rf = rf_model_class.predict(x_test_class)
acc_class_rf = accuracy_score(y_test_class, y_pred_class_rf)
print("Random Forest Classification Accuracy:", acc_class_rf * 100)

# Preparing data for regression (Linear Regression, Polynomial Regression, Decision Tree, Random Forest)
x_reg = np.array(data['Fare']).reshape(-1, 1)
y_reg = np.array(data['Survived'])
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, test_size=0.20, random_state=42)

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(x_train_reg, y_train_reg)
y_pred_reg_lin = lin_model.predict(x_test_reg)
mse_reg_lin = mean_squared_error(y_test_reg, y_pred_reg_lin)
print("Linear Regression Mean Squared Error:", mse_reg_lin)
print(data)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
x_poly_train = poly.fit_transform(x_train_reg)
x_poly_test = poly.transform(x_test_reg)
poly_model = LinearRegression()
poly_model.fit(x_poly_train, y_train_reg)
y_pred_reg_poly = poly_model.predict(x_poly_test)
mse_reg_poly = mean_squared_error(y_test_reg, y_pred_reg_poly)
print("Polynomial Regression Mean Squared Error:", mse_reg_poly)

# Decision Tree
dt_model_reg = DecisionTreeRegressor()
dt_model_reg.fit(x_train_reg, y_train_reg)
y_pred_reg_dt = dt_model_reg.predict(x_test_reg)
mse_reg_dt = mean_squared_error(y_test_reg, y_pred_reg_dt)
print("Decision Tree Regression Mean Squared Error:", mse_reg_dt)

# Random Forest
rf_model_reg = RandomForestRegressor()
rf_model_reg.fit(x_train_reg, y_train_reg)
y_pred_reg_rf = rf_model_reg.predict(x_test_reg)
mse_reg_rf = mean_squared_error(y_test_reg, y_pred_reg_rf)
print("Random Forest Regression Mean Squared Error:", mse_reg_rf)

# Plotting Linear Regression
plt.scatter(x_test_reg, y_test_reg, color='blue')
plt.plot(x_test_reg, y_pred_reg_lin, color='red', linewidth=2)
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.title('Linear Regression - Fare vs Survived')
plt.show()

# Plotting Logistic Regression
plt.scatter(x_test_class, y_test_class,color= 'blue')
plt.plot(x_test_class, y_pred_class_log,color= 'blue',linewidth=2)
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.title('Logistic Regression - Sex vs Survived')
plt.show()

# Plotting Polynomial Regression
plt.scatter(x_test_reg, y_test_reg, color='blue')
plt.plot(x_test_reg, y_pred_reg_poly, color='green', linewidth=2)
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.title('Polynomial Regression - Fare vs Survived')
plt.show()

# K-Means Clustering
df = pd.DataFrame()
df['Sex'] = data['Sex']
df['Fare'] = data['Fare']
df = np.array(df)
kmeans = KMeans(n_clusters=2)
kmeans.fit(df)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = ['orange', 'blue']

plt.figure(figsize=(8, 6))

for i in range(len(df)):
    plt.scatter(df[i, 0], df[i, 1], c=colors[labels[i]], s=50, alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='.', label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.legend()

for idx, centroid in enumerate(centroids):
    centroid_str = ", ".join(map(str, centroid))
    print(f"Cluster {idx + 1}: {centroid_str}")

plt.show()