import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

#data set
df = pd.read_csv(r"C:\Users\gnane\Downloads\cardekho.csv")
print(df)

print("First 5 rows:",df.head())
print("descriptive statistics for numerical column:\n",df.describe())
print("Information about datsert:\n",df.info())
print("checking Null Values:\n",df.isnull())
print("Sum of null values:\n",df.isnull().sum())
print(df.isnull().sum().sum())
print("Correlation:\n",df.corr)
print("Last 5 rows:\n",df.tail())

df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

# Calculate the median for the columns with null values
mileage_median = df['mileage(km/ltr/kg)'].median()
engine_median = df['engine'].median()
max_power_median = df['max_power'].median()
seats_median = df['seats'].median()

df['mileage(km/ltr/kg)'] = df['mileage(km/ltr/kg)'].fillna(mileage_median)
df['engine'] = df['engine'].fillna(engine_median)
df['max_power'] = df['max_power'].fillna(max_power_median)
df['seats'] = df['seats'].fillna(seats_median)

print(df.isnull().sum())

#relationship between actual car selling prices and prices predicted by the KNN model.
r = df[['year','km_driven']]
t = df['selling_price']

# Train-test split
r_train, r_test, t_train, t_test = train_test_split(
    r, t, test_size=0.2, random_state=42
)

# Feature scaling (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(r_train)
X_test_scaled = scaler.transform(r_test)

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train model
knn.fit(X_train_scaled, t_train)

# Predictions
u_pred = knn.predict(X_test_scaled)
print("Accuracy_score:\n",accuracy_score(u_pred,t_test))
print("Confusion_matrix:\n",confusion_matrix(u_pred,t_test))

plt.figure(figsize=(8, 6))
plt.scatter(t_test,u_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("KNN Regression Scatter Plot – CarDekho Dataset")

# Reference line (perfect prediction)
plt.plot(
    [t_test.min(), t_test.max()],
    [t_test.min(), t_test.max()],
)
plt.show()

#Multinomial Naive Bayes model classifies car transmission types (Manual vs Automatic) based on car names.
a = df['name']
b = df['transmission']

vec = CountVectorizer()
a_vec = vec.fit_transform(a)

a_train,a_test,b_train,b_test = train_test_split(a_vec,b,random_state=42,test_size=0.2)

MNB = MultinomialNB()
MNB.fit(a_train,b_train)

b_pred = MNB.predict(a_test)
b_proba = MNB.predict_proba(a_test)

print("Probability:\n",b_proba)
print("Accuracy_score:\n",accuracy_score(b_pred,b_test))

CM = confusion_matrix(b_pred,b_test)
print("Confusion_Matrix:\n",CM)
# heat map
plt.figure(figsize=(8,6))
sns.heatmap(CM,annot=True,cmap='coolwarm',fmt ='d',
            xticklabels=MNB.classes_,
            yticklabels=MNB.classes_)
plt.xlabel("Predicted values of car names")
plt.ylabel("Actual values of the Transmission")
plt.title("Car Transmission Mapping: Manual vs Automatic Heatmap")
plt.show()


#decision tree rules used to classify fuel types based on mileage and engine power.
x = df[['mileage(km/ltr/kg)','max_power']]
y = df['fuel']

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

dtree = DecisionTreeClassifier(max_depth=3,criterion='entropy',random_state=42)
dtree.fit(x_train,y_train)

y_pred = dtree.predict(x_test)
y_proba = dtree.predict_proba(x_test)

print("Accuracy_score:\n",accuracy_score(y_pred,y_test))
CM = confusion_matrix(y_pred,y_test)
print("Confusion_matrix:\n",CM)

plt.figure(figsize=(8,6))
plot_tree(dtree,class_names=dtree.classes_,feature_names=x.columns,filled=True,fontsize=8)
plt.title("Decision Tree Classifier for Fuel Type Prediction")
plt.show() 

# multionomail regression on predictive interval plot
features = [
    'km_driven',
    'mileage(km/ltr/kg)',
    'engine',
    'max_power'
]
target = 'selling_price'
X = df[features]
Y = df[target]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, Y_train)

o_pred = model.predict(X_test_scaled)

n = X_train_scaled.shape[0]  #number of training samples
p = X_train_scaled.shape[1]  #number of input features

# Residual standard error  measures prediction error
residuals = Y_train - model.predict(X_train_scaled)
s_err = np.sqrt(np.sum(residuals**2) / (n - p - 1))

# t value for 95% confidence
t_val = stats.t.ppf(0.975, n - p - 1)

# Prediction interval
pred_interval = t_val * s_err

lower_bound = o_pred - pred_interval
upper_bound = o_pred + pred_interval

plt.figure(figsize=(8, 5))

x_axis = np.arange(len(y_test))

plt.plot(x_axis, Y_test.values)
plt.plot(x_axis, o_pred)
plt.fill_between(x_axis, lower_bound, upper_bound, alpha=0.3)

plt.xlabel("Test Samples")
plt.ylabel("Selling Price")
plt.title("Prediction Interval Plot for Car Selling Price")

plt.show()

#QUADRANT Bubble  PLOT on  K-means clutering

df['Business_Value'] = (
    (df['selling_price'] - df['selling_price'].min()) /
    (df['selling_price'].max() - df['selling_price'].min())
) * 10

# Technical Value (Power + Condition)
df['Technical_Value'] = (
    (df['max_power'] / df['max_power'].max()) * 5 +
    (1 - (df['km_driven'] / df['km_driven'].max())) * 5
)

# Bubble Size
df['Bubble_Size'] = df['selling_price'] / df['selling_price'].max() * 2500

X = df[['Business_Value', 'Technical_Value']]

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

cluster_colors = {
    0: '#ff006e',   
    1: '#3a86ff',   
    2: '#06d6a0',   
    3: '#ffd166'    
}
df['Color'] = df['Cluster'].map(cluster_colors)

plt.figure(figsize=(9, 7))

plt.scatter(
    df['Business_Value'],
    df['Technical_Value'],
    s=df['Bubble_Size'],
    c=df['Color'],
    alpha=0.75,
    edgecolors='black'
)

# Quadrant reference lines
plt.axvline(5, linestyle='--')
plt.axhline(5, linestyle='--')

# Cluster centers
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=300,
    c='black',
    marker='X',
    label='Cluster Centers'
)

# Labels
plt.xlabel("Business Value", fontsize=11)
plt.ylabel("Technical Value", fontsize=11)
plt.title("K-Means Based Car Valuation Quadrant Bubble Chart", fontsize=13)

plt.xlim(0, 10)
plt.ylim(0, 10)

# Quadrant text
plt.text(7.5, 8.5, "High Value\nLow Risk", ha='center')
plt.text(2, 8.5, "Low Value\nLow Risk", ha='center')
plt.text(2, 2, "Low Value\nHigh Risk", ha='center')
plt.text(7.5, 2, "High Value\nHigh Risk", ha='center')

plt.grid(alpha=0.3)
plt.legend()
plt.show()

#To reduce multiple car-related numeric features into principal components using PCA
numeric_cols = [
    'selling_price',
    'km_driven',
    'mileage(km/ltr/kg)',
    'engine',
    'max_power'
]

df_numeric = df[numeric_cols]

# Step 2: Scale the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Step 3: Apply PCA
pca = PCA(n_components=5)
pca_values = pca.fit_transform(df_scaled)

# Step 4: Take mean PCA values (ALL cars)
mean_pca_values = np.mean(pca_values, axis=0)

# Labels for radar chart
labels = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

# Close the radar chart loop
values = np.append(mean_pca_values, mean_pca_values[0])
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
angles = np.append(angles, angles[0])

# Step 5: Plot radar chart
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.3)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Radar Chart of All Cars using PCA (Average Profile)")

plt.show()

