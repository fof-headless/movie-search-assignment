# Import libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import hashlib
import os
import tqdm as notebook_tqdm


# I have used hashing to check if the CSV file has changed since last embedding computation. If it has changed, new embeddings will be computed and cached.

def get_file_hash(filepath):

    # Calculate MD5 hash of a file
    
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def search_movies(query, top_n=5, csv_path='movies.csv', embeddings_dir='embeddings'):


    """
    Search for movies based on query similarity to plot descriptions
    Creates and caches embeddings for movie plots to speed up repeated searches and generate embedding once.
    
    Args:
        query (str): Search query describing the type of movie/plot
        top_n (int): Number of top similar movies to return

        OPTIONAL QUERIES:

        csv_path (str): Path to movies CSV file
        embeddings_dir (str): Directory to store embedding cache
    
    Returns:
        pandas.DataFrame: Top N most similar movies with similarity scores
    """


    # Create the embeddings dir if it does not exist

    os.makedirs(embeddings_dir, exist_ok=True)
    

    # Calculate the CSV hash

    current_hash = get_file_hash(csv_path)
    hash_file = os.path.join(embeddings_dir, 'csv_hash.txt')
    embeddings_file = os.path.join(embeddings_dir, 'plot_embeddings.pkl')
    

    # Check if cached embeddings exist and CSV hasn't changed

    use_cache = False
    if os.path.exists(hash_file) and os.path.exists(embeddings_file):
        with open(hash_file, 'r') as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            use_cache = True
    
    # Load dataset

    df = pd.read_csv(csv_path)
    


    # Load the Sentence Transformer model

    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if use_cache:
        # Load cached embeddings
        with open(embeddings_file, 'rb') as f:
            plot_embeddings = pickle.load(f)
    else:
        # Create new embeddings
        plot_embeddings = model.encode(df['plot'].tolist(), show_progress_bar=True)
        
        # Cache the embeddings and hash
        with open(embeddings_file, 'wb') as f:
            pickle.dump(plot_embeddings, f)
        with open(hash_file, 'w') as f:
            f.write(current_hash)
    

    # Encode the query

    query_embedding = model.encode([query])
    

    # Calculate cosine similarity between query and all movie plots

    similarities = cosine_similarity(query_embedding, plot_embeddings)[0]
    

    # Get indices of top N most similar movies

    top_indices = np.argsort(similarities)[::-1][:top_n]

    
    # Create result dataframe

    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    

    # Sort by similarity score (descending)

    results = results.sort_values('similarity', ascending=False)
    


    return results[['title', 'plot', 'similarity']]