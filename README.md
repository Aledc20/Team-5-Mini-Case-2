# Team-5-Mini-Case-2
import seaborn as sns
import pandas as pd

# Include Any Additional Packages you need to import


# Load the "mpg" dataset from seaborn's GitHub repository
df = sns.load_dataset('mpg')

# Alternatively, you can load the "mpg" dataset directly from the UCI Machine Learning Repository:
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', delim_whitespace=True, header=None, names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Print the first few rows of the dataset to verify that it's loaded correctly
df.head()

[ ]
df.describe()

[ ]
plt.scatter(x=df.horsepower ,y=df.weight)

[ ]
import matplotlib.pyplot as plt
df.hist("weight")

[ ]
import seaborn as sns
sns.pairplot(df)

Based on the calculations done. mpg and weight are the best candidates for linear regression.

[ ]
df.corr()

[ ]
sns.heatmap(df.corr())

There are different linear relations between MPG and weight, for example and also other variables like acceleration and mpg and acceleration and model_year

[ ]
X = df.weight
y = df.mpg
[ ]
from sklearn.linear_model import LinearRegression
[ ]
model = LinearRegression()
[ ]
import statsmodels.api as sm
model=sm.OLS(y,X).fit()
model.predict(X)
0      23.641070
1      24.916231
2      23.182282
3      23.162042
4      23.269992
         ...    
393    18.823797
394    14.370856
395    15.484091
396    17.710562
397    18.351516
Length: 398, dtype: float64
