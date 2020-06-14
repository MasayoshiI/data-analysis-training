import pandas as pd
from sklearn.datasets import load_iris

class IrisAnalysis:
    """ Analysis for Iris"""
    def __init__(self, dataset):
        self.df = pd.DataFrame(dataset)

    def get_desc(self):
        return self.df.describe()

    


if __name__ == "__main__":
    # initialize iris dataset as iris
    iris = load_iris()
    # show description
    # print(iris.DESCR)
    # Shape
    print(iris.data.shape)
    # type of flowers
    print(iris.target_names)
    # feature names
    print(iris.feature_names)
