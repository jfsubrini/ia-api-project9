# -*- coding: utf-8 -*-
"""
Created by Jean-François Subrini on the 17th of March 2023.
Creation of a simple sentiment analysis REST API 
using the FastAPI framework and a Content-Based approach (created in the Jupyter Notebook).
This REST API has been deployed on Azure Web App (https://ia-project9.azurewebsites.net).  TODO
"""
# Importation of libraries.
import pandas as pd
import uvicorn
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity


# Creating the app object.
app = FastAPI()

# Creating DataFrames, loading .csv files.
df_cb = pd.read_csv('df_cb.csv', sep=',', low_memory=False)
df_embed = pd.read_csv('df_embed.csv', sep=',', low_memory=False)

### UTIL FUNCTIONS ###
def mean_embeddings(art_list):
    """Function to calculate the mean embeddings with a list of article ids.
    """
    # Selecting the only embeddings of the articles list.
    df_emb = df_embed.iloc[art_list, :]

    # Calculating the mean embeddings, the synthesis of all articles.
    mean_emb = df_emb.mean()

    # Transpose the result into a dataframe with one row (article)
    # and 250 columns (embedding vector).
    mean_emb = mean_emb.to_frame().transpose()

    return mean_emb

def cb_recommender(user_id, top_n=5):
    """Function to return the top N articles recommendation for an user id,
    using a Content-Based filtering Recommender System, based on embeddings.
    The returned list, by default 5 articles ids, is sorted by confidence descending.
    """
    # Creating a list of article ids with the read articles by the user.
    user_art_list = df_cb.loc[
        df_cb['user_id'] == user_id]['article_id'].to_list()

    # Calculating the mean of these read articles embeddings,
    # weighted by the number of clicks on each article, creating a 'synthetic article'.
    mean_emb = mean_embeddings(user_art_list)

    # Calculating cosine similarity between the 'synthetic article' and
    # all the other articles, with the embeddings vectors.
    # Results into a dataframe, excluding the already read articles.
    cosine_sim = cosine_similarity(df_embed, mean_emb)
    df_cosine_sim = pd.DataFrame(cosine_sim, columns=['cosine_sim'])
    df_cosine_sim = df_cosine_sim.drop(index=user_art_list)

    # Selecting the N closest articles, the ones with higher cosine.
    top_n_art_sim = df_cosine_sim.sort_values(
        by='cosine_sim', ascending=False).head(top_n)
    top_n_articles_list = top_n_art_sim.index.tolist()

    return top_n_articles_list
###---###

# Index route, opens automatically on http://127.0.0.1:8000.
# after typing 'uvicorn main:app --reload' in the Terminal.
# or http://127.0.0.1:8080 with 'uvicorn main:app --reload --port 8080' if Django client.
@app.get('/')
def index():
    """Welcome message"""
    return {'message': 'This is a content-based firltering recommender system app.'}

# Route with a selected user id parameter, returns the 5 articles recommendation for that user.
# Located at: http://127.0.0.1:8000/recommender/?select_user_id=
# Also access to the FastAPI swagger to type directly the user id to get a recommendation.
# Located at: http://127.0.0.1:8000/docs
@app.get('/recommender/')
def recommender(select_user_id: str):
    """Get a 5 articles recommendation for a selected user id."""
    # 5 articles recommendation list for a specific user id.
    print("select_user_id API : ", select_user_id, type(select_user_id))
    reco = cb_recommender(int(select_user_id))
    print("RECO API : ", reco)
    return {'reco': reco}


# Running the API with uvicorn.
# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
