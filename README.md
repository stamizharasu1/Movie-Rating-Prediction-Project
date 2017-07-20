# Movie-Rating-Prediction-Project
Given a set of training data, This is a project written in Python that predicts a set of movie ratings for users, based on the movies they liked.

The given training data is first read in and stored using numpy arrays.

Each of the test data files contain users, and a set of movies that they have given ratings for. We are then tasked with predicting what rating they will give for the movies that have a blank space next to them.

In order to predict these ratings, we use a variety of algorithms, including  user and item based Cosine Similarity, Pearson Correlation, and Inverse Document Frequency. We calculate the MAE of each of these sets of predictions to see which algorithm gave us the most accurate results. 
