
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

# ### Basic linear regression model with basic numeric data (no categorical or dummy data)
df = pd.read_csv('../data/cleaned_data.csv')

X = df.drop(columns = ['saleprice', 'id'])._get_numeric_data()
y = df[['saleprice']]

lr = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr.fit(X_train, y_train)
base_score = lr.score(X_train, y_train), lr.score(X_test, y_test), cross_val_score(lr, X_train, y_train).mean()
base_score

preds = lr.predict(X_test)

residuals = y_test - preds

# %%
plt.figure(figsize=(10,6))
plt.hist(residuals, bins=20)
plt.axvline(0, color='red')
plt.title('LR model Residuals Distribution', fontsize=15)
plt.xlabel('Residuals', fontsize=12)
plt.savefig('../images/num_model_residuals.png', bbox_inches='tight');



plt.figure(figsize=(10,6))
plt.scatter(preds, residuals, alpha=0.5)
plt.title('Scatter of Predictions vs Residuals', fontsize=15)
plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.axhline(0, color='red')
plt.savefig('../images/num_model_scatter.png', bbox_inches='tight');

# ### Basic linear regression model with added numerized categorical data 

df_num = pd.read_csv('../data/cleaned_numerized_data.csv')

X_num = df_num.drop(columns = ['saleprice', 'id'])._get_numeric_data()
y_num = df_num[['saleprice']]

lr_num = LinearRegression()
X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(X_num, y_num, random_state=42)
lr_num.fit(X_num_train, y_num_train)
num_scores = lr_num.score(X_num_train, y_num_train), lr_num.score(X_num_test, y_num_test), cross_val_score(lr_num, X_num_train, y_num_train).mean()

num_preds = lr_num.predict(X_num_test)
num_residuals = y_num_test - num_preds

plt.figure(figsize=(10,6))
plt.hist(num_residuals, bins=20)
plt.axvline(0, color='red')
plt.title('LR model Residuals Distribution', fontsize=15)
plt.xlabel('Residuals', fontsize=12)
plt.savefig('../images/cat_model_residuals.png', bbox_inches='tight');


plt.figure(figsize=(10,6))
plt.scatter(num_preds, num_residuals, alpha=0.5)
plt.title('Scatter of Predictions vs Residuals', fontsize=15)
plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.axhline(0, color='red')
plt.savefig('../images/cat_model_scatter.png', bbox_inches='tight');

# With scores of around 89 percent, the basic numeric data model in its current stage of complexity explains given percentage of sale prices. 

# ### Model with interaction parameters and dummies added

df_all = pd.read_csv('../data/dummy_numerize_interact_data.csv')

X_all = df_all.drop(columns = ['saleprice', 'id'])._get_numeric_data()
y_all = df_all[['saleprice']]

lr_all = LinearRegression()
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, random_state=42)
lr_all.fit(X_all_train, y_all_train)
all_scores = lr_all.score(X_all_train, y_all_train), lr_all.score(X_all_test, y_all_test), cross_val_score(lr_all, X_all_train, y_all_train).mean()

all_preds = lr_all.predict(X_all_test)
all_residuals = y_all_test - all_preds


plt.figure(figsize=(10,6))
plt.hist(all_residuals, bins=20)
plt.axvline(0, color='red')
plt.title('LR model Residuals Distribution', fontsize=15)
plt.xlabel('Residuals', fontsize=12)
plt.savefig('../images/all_model_residuals.png', bbox_inches='tight');


plt.figure(figsize=(10,6))
plt.scatter(num_preds, num_residuals, alpha=0.5)
plt.title('Scatter of Predictions vs Residuals', fontsize=15)
plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.axhline(0, color='red')
plt.savefig('../images/all_model_scatter.png', bbox_inches='tight');

print(all_scores, num_scores, base_score)

# As seen from the scores above, the model only benefited from additional details introduced with each step. The final model has ~93% accuracy.