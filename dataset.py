# Imports
import pandas as pd

# Dataset Load
dataset = pd.read_csv('wiki_movie_plots_deduped.csv')

# Retain the movie and plot only
dataset = dataset[['title', 'plot']]

#Export the cleaned dataset
dataset.to_csv('movies.csv', index=False)