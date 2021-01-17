import pandas as pd
import numpy as np

movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'
similarity_file = "./matrices/item_item_cf.csv"

movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])
similarity = pd.read_csv(similarity_file, delimiter=',', header=None)

def item_item_cf_with_baseline(movies, users, ratings, predictions):
    data = ratings.pivot_table(index='movieID', values='rating',
                               columns='userID')
    matrix = pd.DataFrame(data=data, columns=users['userID'],
                          index=movies['movieID'])

    global_mean = ratings["rating"].mean()

    users_mean = matrix.mean(axis=0).fillna(global_mean)
    user_deviation = users_mean - global_mean

    movies_mean = matrix.mean(axis=1).fillna(global_mean)
    movies_deviation = movies_mean - global_mean

    global_baselines = pd.DataFrame(global_mean, index=movies["movieID"], columns=users["userID"])

    global_baselines = global_baselines.transpose().add(movies_deviation).transpose()

    global_baselines = global_baselines.add(user_deviation)

    mean = ratings.pivot_table(index='movieID', values='rating').rename(
        columns={"rating": "mean"})
    merged = pd.merge(ratings, mean, on='movieID')
    merged['rating'] -= merged['mean']
    merged = merged.drop(columns=['mean'])
    norm_data = merged.pivot_table(index='movieID', values='rating',
                                   columns='userID')
    normalized_matrix = pd.DataFrame(data=norm_data, index=movies['movieID'],
                                     columns=users['userID']).fillna(0).astype(
        'float64')

    # similarity = calculate_similarity_matrix(normalized_matrix)

    sim_matrix = similarity

    sim_matrix.index = normalized_matrix.index
    sim_matrix.columns = normalized_matrix.index
    np_sim_matrix = sim_matrix.to_numpy()
    np_sim_matrix = np_sim_matrix + np_sim_matrix.T - np.diag(
        np.diag(np_sim_matrix))

    sim_matrix = pd.DataFrame(np_sim_matrix, index=sim_matrix.index,
                              columns=sim_matrix.columns)


    k = 15

    submission = []

    def get_prediction(entry):
        user = entry[0]
        movie = entry[1]

        gb = global_baselines.loc[movie, user]

        movie_ratings = matrix[user].dropna()
        k_similar_movies_similarities = sim_matrix[movie][movie_ratings.index].sort_values(
            ascending=False).head(k)
        k_similar_movies_indices = k_similar_movies_similarities.index
        k_similar_movies_ratings = movie_ratings[k_similar_movies_indices] - global_baselines.loc[k_similar_movies_indices, user]

        prediction = gb
        if k_similar_movies_similarities.sum() != 0 and len(k_similar_movies_similarities) > 0:
            prediction += np.average(k_similar_movies_ratings, weights=k_similar_movies_similarities,
                                    axis=0)

        submission.append(prediction)

    print("Generating predictions")

    predictions.apply(get_prediction, axis=1)

    submission = list(enumerate(submission, 1))

    return submission

predictions = item_item_cf_with_baseline(movies_description, users_description, ratings_description, predictions_description)

with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)
