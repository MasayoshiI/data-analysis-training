import pandas as pd
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
import seaborn as sns
# import sys


bsdata = load_boston()
df = pd.DataFrame(bsdata.data, columns=bsdata.feature_names) 
df['Price'] = bsdata.target
# print(df)

# obtain correlations
corr = df.corr()
# color dataset
corr.style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(corr)


# low income ratio
sns.jointplot('LSTAT', 'Price', data=df)

# criminal rate n price
sns.jointplot('CRIM', 'Price', data=df, kind="reg")


plt.show()