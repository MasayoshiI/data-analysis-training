import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()

print(iris.data.shape)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.loc[df['target'] == 0, 'target'] = "setosa"
df.loc[df['target'] == 1, 'target'] = "versicolor"
df.loc[df['target'] == 2, 'target'] = "virginica"
print(df.describe())

### Pairplot to see the correlation of every possible pair of features
sns.pairplot(df, hue="target")
# plt.show()

# Set X and y
X = iris.data[:, [0, 2]] 
y = iris.target

# graph common settings
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
