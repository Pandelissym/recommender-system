import numpy as np
import pandas as pd
import os.path
from random import randint
import time

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'
similarity_file = 'matrices/user_user_cf_similarity.csv'
# Read the data using pandas

movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)
# similarity = pd.read_csv(similarity_file, delimiter=',', header=None)
#####
##
## COLLABORATIVE FILTERING
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
    print("Creating normalized utility matrix")
    data = ratings.pivot_table(index='userID', values='rating', columns='movieID')
    matrix = pd.DataFrame(data=data, index=users['userID'], columns=movies['movieID'])

    mean = ratings.pivot_table(index='userID', values='rating').rename(columns={"rating": "mean"})
    merged = pd.merge(ratings, mean, on='userID')
    merged['rating'] -= merged['mean']
    merged.drop(columns=['mean'])
    norm_data = merged.pivot_table(index='userID', values='rating', columns='movieID')
    normalized_matrix = pd.DataFrame(data=norm_data, index=users['userID'], columns=movies['movieID']).fillna(0).astype('float64')

    print(normalized_matrix)

    start = time.perf_counter()

    print("Calculating lengths")

    lengths = normalized_matrix.apply(np.linalg.norm, axis=1).to_numpy()
    print(lengths)

    print("Creating similarity matrix")

    normalized_matrix = normalized_matrix.to_numpy()
    sim_matrix = np.ndarray((normalized_matrix.shape[0], normalized_matrix.shape[0]))

    for i, x in enumerate(normalized_matrix):
        for j, y in enumerate(normalized_matrix):
            if i <= j:
                sim_matrix[i][j] = np.dot(x, y) / (lengths[i] * lengths[j])

    end = time.perf_counter()
    print("Time taken: ", round(end - start), " seconds")

    with open(similarity_file, 'w') as sim_writer:
        sim = [map(str, row) for row in sim_matrix]
        sim = [','.join(row) for row in sim]
        sim = '\n'.join(sim)
        sim_writer.write(sim)

    print(sim_matrix)

    sim_matrix = similarity
    print(sim_matrix.head())

    sim_matrix.index = normalized_matrix.index
    sim_matrix.columns = normalized_matrix.index
    np_sim_matrix = sim_matrix.to_numpy()
    np_sim_matrix = np_sim_matrix + np_sim_matrix.T - np.diag(np.diag(np_sim_matrix))

    sim_matrix = pd.DataFrame(np_sim_matrix, index=sim_matrix.index, columns=sim_matrix.columns)
    k = 3

    submission = []

    def get_prediction(entry):
        user = entry[0]
        movie = entry[1]

        neighbor_ratings = matrix[movie].dropna()
        neighbours_sim = sim_matrix[user][neighbor_ratings.index].sort_values(ascending=False).head(k)
        neighbor_ratings = neighbor_ratings[neighbours_sim.index]

        prediction = 0
        if neighbours_sim.sum() != 0 and len(neighbours_sim) > 0:
            prediction = np.average(neighbor_ratings, weights=neighbours_sim, axis=0)

        submission.append(prediction)

    predictions.apply(get_prediction, axis=1)

    submission = list(enumerate(submission, 1))

    return submission


#####
##
## LATENT FACTORS
##
#####
    
def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass
    
    
#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
  ## TO COMPLETE

  pass


#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####
    
#By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]

#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function

# predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)
predictions = predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)


# #Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)

    #Writes it dowmn
    submission_writer.write(predictions)