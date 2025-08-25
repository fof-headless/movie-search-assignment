import unittest
import pandas as pd
import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer

# Ensure project root is on sys.path so that import movie_search does not result in an error
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from movie_search import search_movies


class TestMovieSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Creates a sample dataset and save to movies.csv in project root
        cls.df = pd.DataFrame({
            'title': ['Spy Movie', 'Romance in Paris', 'Action Flick'],
            'plot': [
                'A spy navigates intrigue in Paris to stop a terrorist plot.',
                'A couple falls in love in Paris under romantic circumstances.',
                'A high-octane chase through New York with explosions.'
            ]
        })
        cls.csv_path = os.path.join(PROJECT_ROOT, "movies.csv")
        cls.df.to_csv(cls.csv_path, index=False)  # <-- write to project root

        # Preload model + embeddings (optional, for speed check)
        cls.model = SentenceTransformer('all-MiniLM-L6-v2')
        cls.embeddings = cls.model.encode(cls.df['plot'].tolist(), convert_to_tensor=False)

    @classmethod
    def tearDownClass(cls):
        # Clean up generated CSV after tests (optional)
        if os.path.exists(cls.csv_path):
            os.remove(cls.csv_path)

    def test_search_movies_output_format(self):
        """Test if search_movies returns a DataFrame with correct columns."""
        query = "spy thriller in Paris"
        result = search_movies(query, top_n=3)
        self.assertIsInstance(result, pd.DataFrame, "Result should be a pandas DataFrame")
        expected_columns = ['title', 'plot', 'similarity']
        self.assertTrue(all(col in result.columns for col in expected_columns),
                        f"Result should have columns: {expected_columns}")

    def test_search_movies_top_n(self):
        """Test if search_movies returns the correct number of results."""
        top_n = 2
        query = "spy thriller in Paris"
        result = search_movies(query, top_n=top_n)
        self.assertEqual(len(result), top_n, f"Result should return {top_n} movies")

    def test_search_movies_similarity_range(self):
        """Test if similarity scores are between 0 and 1."""
        query = "spy thriller in Paris"
        result = search_movies(query, top_n=3)
        similarities = result['similarity'].values
        self.assertTrue(all(0 <= sim <= 1 for sim in similarities),
                        "Similarity scores should be between 0 and 1")

    def test_search_movies_relevance(self):
        """Test if returned movies are relevant to the query."""
        query = "spy thriller in Paris"
        result = search_movies(query, top_n=1)
        top_plot = result.iloc[0]['plot'].lower()
        self.assertTrue(any(term in top_plot for term in ['spy', 'thriller', 'paris']),
                        "Top result should relate to query terms")


if __name__ == '__main__':
    unittest.main()