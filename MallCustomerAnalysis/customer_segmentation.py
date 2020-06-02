# Author: Masayoshi Iwasa
# Desc: to practice basic plt skills and k-mean clustering 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm

# from pd.tools import plotting

class MallCustAnalyzer():
    """Mall Customer Data Analyze Class"""

    def __init__(self, filename, k):
        """Constructor for MallCustAnalyzer with param filename"""

        self.filename = filename
        self.df = self.get_df()
        self.k = k
        self.X_norm = self.normalize_data()
        self.kmean_model = self.get_kmean(self.k)
    
    def get_filename(self):
        """get file name """

        return self.filename
    
    def get_df(self):
        """creates df from file"""

        return pd.read_csv(self.filename)

    def get_summary(self):
        """get summary of df"""

        return self.df.describe()
    
    def plot_bar_graph(self, column):
        """plot a bar graph for given column"""

        d = self.df [column].value_counts().to_dict()
        plt.bar(range(len(d)), list(d.values()), align='center')
        plt.xticks(range(len(d)), list(d.keys()))
        plt.title(column)
        plt.show()
    
    def draw_pie_chart(self, column):
        """create pie chart for a given column"""

        d = self.df [column].value_counts().to_dict()
        plt.pie(list(d.values()), labels = list(d.keys()),autopct="%.1f%%")
        # plt.xticks(range(len(d)), list(d.keys()))
        plt.title(column)
        plt.show()

    # def plot_histogram(self, column, bins):
    #     """create hisogram for a given column"""
    #     d = self.df [column].value_counts().to_dict()
    #     # df = pd.DataFrame({"a": np.random.random_integers(0, high=100, size=100)})

    #     ranges = [0,10,20,30,40,50,60,70,80,90,100]
    #     df.groupby(pd.cut(df.a, ranges)).count()
    #     plt.hist(list(d.values()), range=[])
    #     # plt.xticks(range(len(d)), list(d.keys()))
    #     print(d)
    #     plt.title(column)
    #     plt.show()
    
    def draw_boxplot(self, column):
        """Draw boc plot for a given column"""
        d = self.df [column].value_counts().to_dict()
        plt.boxplot(list(d.keys()))
        # plt.xticks(range(len(d)), list(d.keys()))
        plt.title(column)
        plt.show()

    def normalize_data(self):
        # Data preprocessing
        X=self.df.iloc[:, 3:5]
        sc=preprocessing.StandardScaler()
        sc.fit(X)
        X_norm=sc.transform(X)

        return X_norm

    def get_kmean(self, k):
        # Cluster the data
        cls = KMeans(n_clusters=k)
        result = cls.fit(self.X_norm)
        
        return result

    def scat_plot_k_mean(self):
        """Apply K-mean clustering to obtain customer segmentation"""
        # Plot the data
        plt.scatter(self.X_norm[:,0],self.X_norm[:,1], c=self.kmean_model.labels_)
        plt.scatter(self.kmean_model.cluster_centers_[:,0],self.kmean_model.cluster_centers_[:,1],s=250, marker='*',c='red')
        plt.title(f'Scatter plot of {self.k} clusters for customer segmentation')
        plt.show()
        
    
    def elbow_method(self):
        """Appliy elbow method to see the optimal value of clustering within the range of k"""
        see = []
        
        # get SSE score for each k 
        for i in np.arange(1, self.k+1):
            see.append(KMeans(i).fit(self.df.iloc[:, 3:5]).inertia_)
        
        # plot SSE score
        plt.plot(np.arange(1, self.k+1), see, 'o-')
        plt.xlabel("Number of Clusters")
        plt.ylabel("Total intra-clusters sum of squares")

        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    def draw_silhouette(self):
        """Apply silhouette method to visually see the optimal value for k"""
        # initiate k mean model 
        cls = KMeans(n_clusters=self.k)

        # predict the cluster number for each cluster
        y_km = cls.fit_predict(self.X_norm)
        cluster_labels = np.unique(y_km)
        n_clusters=cluster_labels.shape[0]

        # compute silhouette value with sample data, cluster number, euclid distance
        silhouette_vals = silhouette_samples(self.X_norm,y_km,metric='euclidean') 
        y_ax_lower, y_ax_upper= 0,0
        yticks = []


        for i,c in enumerate(cluster_labels):
                c_silhouette_vals = silhouette_vals[y_km==c]      # cluster_labelsには 0,1,2が入っている（enumerateなのでiにも0,1,2が入ってる（たまたま））
                c_silhouette_vals.sort()
                # set max val for y axis by adding up the number of samples per cluster
                y_ax_upper += len(c_silhouette_vals)              
                color = cm.jet(float(i)/n_clusters)               
                plt.barh(range(y_ax_lower,y_ax_upper),            
                                 c_silhouette_vals,               
                                 height=1.0,                      
                                 edgecolor='none',                
                                 color=color)                    
                # set where culster label will be placed at
                yticks.append((y_ax_lower+y_ax_upper)/2)
                y_ax_lower += len(c_silhouette_vals)

        # get avg of silhouette value
        silhouette_avg = np.mean(silhouette_vals)
        # draw - line on avg
        plt.axvline(silhouette_avg,color="red",linestyle="--")
        # cluster level
        plt.yticks(yticks,cluster_labels + 1)                     
        plt.ylabel('Cluster')
        plt.xlabel('silhouette coefficient')
        plt.show()

    
def main():
    # Read/convert to df data using pd 
    cust_analyzer = MallCustAnalyzer("dataset/Mall_Customers.csv",10)
    df = cust_analyzer.df
    print(df)
    
    head = df.head()
    print("Head of cust_data:\n", head)
    print()
    
    summary = df.describe()
    print("Summary of cust_data:\n",summary)
    # print(summary['Spending Score (1-100)'].mean())
    # cust_analyzer.plot_bar_graph("Genre")
    # cust_analyzer.plot_pie_chart("Genre")
    # cust_analyzer.draw_boxplot("Age")
    # cust_analyzer.scat_plot_k_mean()
    # cust_analyzer.elbow_method()
    # cust_analyzer.draw_silhouette()
    
if __name__ == "__main__":  
    main()