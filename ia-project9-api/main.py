# -*- coding: utf-8 -*-
"""
Created by Jean-François Subrini on the 8th of April 2023.
Creation of a simple recommender system REST API using the FastAPI framework 
and a Model-Based Collaborative Filtering approach, with the SVD algorithm, 
created in the Subrini_JeanFrancois_2_scripts_012023.ipynb Jupyter Notebook.
This REST API has been deployed on Heroku (https://ia-api-project9.herokuapp.com).
"""
# Importation of libraries.
from collections import defaultdict
import pickle5 as pickle
import uvicorn
from fastapi import FastAPI


# Creating the app object.
app = FastAPI()


### UTIL FUNCTIONS ###
def get_top_n(predictions, num=5):
    """Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of the SVD algorithm.
        num(int): The number of recommendation to output for each user. Default is 5.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size num. 
        Raw means as in the original dataframe file.
    """
    # Mapping the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sorting the predictions for each user and retrieving the n (num) highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:num]

    return top_n

def prediction_for_user(user_id, num=5):
    """Return the list of 5 (by default, or num value) recommended articles 
    for a specific user id.
    """
    # Loading the prediction model.
    # Opening the prediction file.
    with open('pred_cf', 'rb') as file:
        # Loading information from that file.
        prediction_cf_model = pickle.load(file)
        # Predicting the top-N (num, 5 by default) recommendation articles for each user
        # from a set of predictions that were in the test dataset, NOT in the training dataset.
        top_n = get_top_n(prediction_cf_model, num=num)

    # Creating the recommended article ids list for the user id selected.
    reco_user_list = []
    for i in top_n[user_id]:
        reco_user_list.append(i[0])

    return reco_user_list
###---###

# Index route, opens automatically on http://127.0.0.1:8000.
# after typing 'uvicorn main:app --reload' in the Terminal.
# or http://127.0.0.1:8080 with 'uvicorn main:app --reload --port 8080' if Django client.
@app.get('/')
def index():
    """Welcome message"""
    return {'message': 'This is a model-based collaborative filtering recommender system app.'}

# Route with a selected user id parameter, returns the 5 articles recommendation for that user.
# Located at: http://127.0.0.1:8000/recommender/?select_user_id=
# Also access to the FastAPI swagger to type directly the user id to get a recommendation.
# Located at: http://127.0.0.1:8000/docs
@app.get('/recommender/')
def recommender(select_user_id: str):
    """Get a 5 articles recommendation list for a selected user id."""
    reco = prediction_for_user(int(select_user_id), num=5)

    return {'reco': reco}


# Running the API with uvicorn.
# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
