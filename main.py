import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import pearsonr

df = pd.read_csv('Walmart_sales.csv')
print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

df['Date'] = pd.to_numeric(df['Date'], errors='coerce')

print(df.info())

ndf = df[[ 'Weekly_Sales', 'Holiday_Flag', 'Fuel_Price', 'CPI']]

plt.scatter(ndf['Fuel_Price'], ndf['Weekly_Sales'], color='blue')
plt.xlabel('Fuel_Price')
plt.ylabel("Weekly_Sales")
plt.show()
plt.scatter(ndf['Holiday_Flag'], ndf['Weekly_Sales'], color='blue')
plt.xlabel('Holiday_Flag')
plt.ylabel("Weekly_Sales")
plt.show()
plt.scatter(ndf['CPI'], ndf['Weekly_Sales'], color='blue')
plt.xlabel('CPI')
plt.ylabel("Weekly_Sales")
plt.show()


X = df[['Holiday_Flag', 'Fuel_Price', 'CPI']]  # Features
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

sns.set(style="ticks")
sns.pairplot(df, x_vars=['Holiday_Flag', 'Fuel_Price', 'CPI'], y_vars='Weekly_Sales', kind='reg', plot_kws={'line_kws':{'color':'red'}})

plt.show()
y_pred = model.predict(X_train)
y_pred_mean = model.predict(X_train).mean()

r_squared = model.rsquared
_, p_value = pearsonr(y_train, y_pred)
print("R-squared:", r_squared)
print("p-value:", p_value)
print("Прогноз середнього значення залежної змінної з надійністю 95%:", y_pred_mean)

pred_df = pd.DataFrame({'y_true': y_train, 'y_pred': y_pred})
# зображення довірчого інтервалу
sns.lmplot(x='y_true', y='y_pred', data=pred_df, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5})
plt.xlabel('Фактичні значення')
plt.ylabel('Прогнозовані значення')
plt.title('Довірчий інтервал для теоретичної лінійної парної регресії')
plt.show()