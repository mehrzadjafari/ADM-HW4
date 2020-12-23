
from sklearn.decomposition import TruncatedSVD

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import stem

from nltk.stem import WordNetLemmatizer 
from collections import defaultdict 

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 
from numpy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer
from collections import Counter,OrderedDict

from scipy import stats






def clean_text(text):
    
    """
    The function for cleaning the text files by tokenizing the text, removing the stopwords and non alphanumerical characters.
    
    Input:
    text (string): The raw text file
    
    Returns:
    final_text (string): Cleaned text file
    
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    final_list=[]
    words = word_tokenize(text)
    good_words = []
    for word in words:
        if word.lower() not in stop_words and word.isalpha():
            good_words.append(word.lower())
    list_to_remove = ['b','br','span', 'one' , 'id', 'none' ]   #this is a list from the past homework to remove some other words
    for word in good_words:
        if word not in list_to_remove:
            final_list.append(word)
    final_text = ' '.join(final_list)
    return final_text  # return the cleaned string 





def T_test_score_distribution(cl_1,cl_2):
    
    """
    The function checks the statistical difference between two clusters
    
    Inputs:
    cl_1 (cluster): The first cluster
    cl_2 (cluster): The second cluster
    
    Returns:
    A confirmation string of the comparison between cl_1 and cl_2
    
    
    * For more info check the comments inside the function
    
    """
    
    # input: two cluster dataframes 
    dist_1=cl_1.Score.values #first distribution
    dist_2=cl_2.Score.values #second distribution
    mean_1 = np.mean(dist_1)  # score mean of the first
    mean_2 = np.mean(dist_2)         #score mean of the second 
    var_1 = np.var(dist_1)           #var of the first
    var_2 = np.var(dist_2)           #var of the second
     # get the number of elements 
    N_1 = len(dist_1)  
    N_2 = len(dist_2)
    df= 2*N_1-2. # degrees of fredom
    T_statistics = (mean_1-mean_2)/np.sqrt((var_1/N_1)+(var_2/N_2)) 
    
    p = 1 - stats.t.cdf(T_statistics,df=df) #t value
    
    if(T_statistics>p):    #compare the values
        print('They are statistically different')
    else:
        print('They are statistically equal')
    
        
    t2, p2 = stats.ttest_ind(dist_1,dist_2) #using scipy function to check the result
    
    
    if(t2>p2):    #compare the values
        print('They are statistically different')
    else:
        print('They are statistically equal')
    
    









def euclidian_distance(x,y):
    
    """
    The function computes the distance between two arrays.
    
    Inputs:
    x (array): The first array
    y (array): The second array
    
    Returns:
    The distance between x and y
    
    """
        return  ((x - y) ** 2).sum(axis = 1) 

    

    

class K_means:
    
    def __init__(self, k,max_iterations=500):  # constructor of the class: initialize number of clusters and max iterations
        self.k = k
        self.max_iteration= max_iterations 
    
        
    def clustering(self, data):  #this is the method that creates the cluster 
       
        data = np.array(data).reshape(data.shape[0],data.shape[1]) # transform the dataset into np.array
        self.centroids= data[np.random.choice(range(data.shape[0]), self.k)] #initialize the centroids choosing randomly
        self.pre_centroids=np.zeros((self.k,data.shape[1]))  #initialize the matrix to store the previous values of the centroids 
        self.cluster= np.zeros(len(data)) # initialize the cluster array, each position contains the cluster label of the ith row
        self.counter=0  # store the number of iterations 
        for i in range(self.max_iteration):  # main loop
           
            for index in range(len(data)):  # for each row we have to find the distance from each cluster 
                row= data[index]
                
                
                distance=euclidian_distance(row,self.centroids) #list of distances
                nearest_centroid= np.argmin(distance,axis=0)  # find the index that corresponds to the minimum distance
                self.cluster[index]= nearest_centroid # the index is the centroid label to assign to the row
               
                
            
            for j in range(self.centroids.shape[0]):   # now we have to change the centroids 
                self.pre_centroids[j]= self.centroids[j] # store the previous value of the centroid j
                cluster= data[self.cluster==j]  # find the rows that belong to the jth centroid 
                new_centroid= cluster.mean(axis=0)  #calculate the rows mean 
                self.centroids[j]=new_centroid # change the centroid value
           
            centroid_distances = euclidian_distance(self.pre_centroids, self.centroids)  # calculate the distance between previous and current centroids 
            
            if(sum(centroid_distances) == 0): # if they did not change we can stop the iterations 
                break
            self.counter+=1
 



