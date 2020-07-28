import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return data[data.index == index]["title"].values[0]

def get_index_from_title(title):
	return data[data.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
data=pd.read_csv("movie_dataset.csv")
#print(data.head())

##Step 2: Select Features
features=['keywords','cast','genres','director']
for feature in features:
	data[feature]=data[feature].fillna("")

##Step 3: Create a column in DF which combines all selected features
def combine_features(row):
	return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

data["combined_features"]=data.apply(combine_features,axis=1)

##Step 4: Create count matrix from this new combined column
cv=CountVectorizer()
count_matrix=cv.fit_transform(data["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim=cosine_similarity(count_matrix)
#print(cosine_sim)
movie_user_likes = "Avatar"

## Step 6: Get index of this movie from its title
movie_index=get_index_from_title(movie_user_likes)
similar_movies=list(enumerate(cosine_sim[movie_index]))


## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies=sorted(similar_movies,key=lambda x: x[1],reverse=True)

## Step 8: Print titles of first 50 movies
i=0
for movie in sorted_similar_movies:
	print(get_title_from_index(movie[0]))
	i=i+1
	if i>50:
		break