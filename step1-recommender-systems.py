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
similarity_file = './data/similarity.csv'
# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

#####
##
## COLLABORATIVE FILTERING
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
    print("Creating normalized utility matrix")
    matrix = ratings.pivot_table(index='userID', values='rating', columns='movieID')

    mean = ratings.pivot_table(index='userID', values='rating').rename(columns={"rating": "mean"})
    ratings = pd.merge(ratings, mean, on='userID')
    ratings['rating'] -= ratings['mean']
    ratings.drop(columns=['mean'])
    normalized_matrix = ratings.pivot_table(index='userID', values='rating', columns='movieID').fillna(0)

    print("Creating similarity matrix")
    # matrix.set_index('userID')
    normalized_matrix = normalized_matrix.to_numpy()
    matrix = matrix.to_numpy()
    similarity_matrix = np.zeros((normalized_matrix.shape[0], normalized_matrix.shape[0]))
    start = time.perf_counter()

    for idx, x in enumerate(normalized_matrix):
        for idx2, y in enumerate(normalized_matrix):
            if idx <= idx2:
                similarity_matrix[idx, idx2] = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    end = time.perf_counter()
    print("Time taken: ", round(end - start), " seconds")

    with open(similarity_file, 'w') as sim_writer:
        sim = [map(str, row) for row in similarity_matrix]
        sim = [','.join(row) for row in sim]
        sim = 'Id,Rating\n' + '\n'.join(sim)

        sim_writer.write(sim)

    similarity_matrix = pd.DataFrame(similarity_matrix)
    # print(similarity_matrix)

    # k = 2
    # user = 1
    # movie = 0
    # neighbours = similarity_matrix[user]\
    #             .drop(user)
    #
    # indices_to_drop = []
    # # for idx, neighbour in neighbours.iteritems():
    # #     if np.isnan(matrix[idx][movie]):
    # #         indices_to_drop.append(idx)
    # #
    # # neighbours = neighbours.drop(indices_to_drop).head(k)
    # #
    # # prediction = 0
    # #
    # # if len(neighbours > 0):
    # #
    # #     for idx, neighbour in neighbours.iteritems():
    # #         prediction += neighbour * matrix[idx][movie]
    # #
    # #     prediction  = prediction / neighbours.sum()






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

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it dowmn
    submission_writer.write(predictions)