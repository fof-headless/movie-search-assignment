# Movie Search Engine

A semantic movie search engine that uses sentence transformers and cosine similarity to find movies based on plot descriptions. Built with [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

## Prerequisites

- Python 3.8 or higher
- uv package manager

## Installation

### 1. Install uv

#### macOS and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Alternative: Install via pip
```bash
pip install uv
```

### 2. Clone the Repository
```bash
git clone https://github.com/fof-headless/movie-search-assignment
cd movie-search-assignment
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
```

## Setup and Usage

### Virtual Environment 
While uv can work without virtual environments, you can create one if preferred:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 1. CSV Setup
First, run the database setup script:
```bash
uv run database.py
```

This script will:
- Initialize the csv file
- Create a formatted and cleaned csv

### 2. Run the Application
After the csv is set up, run the main application:
```bash
uv run movie_search.py
```

The application provides a semantic movie search engine that:
- Uses sentence transformers to create embeddings of movie plot descriptions
- Caches embeddings for performance (automatically regenerates if CSV changes)
- Finds movies similar to your search query using cosine similarity
- Returns top matching movies with similarity scores

## Quick Start

For a complete setup from scratch:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and navigate to project
git clone https://github.com/fof-headless/movie-search-assignment
cd movie-search-assignment

# Install dependencies
uv pip install -r requirements.txt

# Setup database
uv run database.py

# Run application
uv run movie_search.py
```

### Interactive Examples
For interactive examples and detailed demonstrations, check out the Jupyter notebook:
```bash
# Install Jupyter (if not already installed)
uv pip install jupyter

# Start Jupyter and open the examples notebook
jupyter notebook movie_search.ipynb
```

The notebook includes:
- Step-by-step usage examples
- Different search query demonstrations  
- Performance comparisons
- Visualization of similarity scores

## Project Structure

```
.
├── README.md
├── requirements.txt
├── database.py          # Database initialization script
├── movie_search.py     # Movie search engine with semantic similarity
├── movies.csv          # Movie dataset with titles and plot descriptions
├── embeddings/         # Cached embeddings directory (auto-created)
│   ├── plot_embeddings.pkl
│   └── csv_hash.txt
├── movie_search.ipynb  # Jupyter notebook with usage examples and demos
└── test/               # Unit tests
    └── test_movie_search.py
```



## Features

- **Semantic Search**: Uses sentence transformers to understand plot context beyond keyword matching
- **Intelligent Caching**: Embeddings are cached and only regenerated when the movie dataset changes
- **Similarity Scoring**: Returns similarity scores to see how closely movies match your query
- **Fast Performance**: Cached embeddings make subsequent searches nearly instantaneous
- **Flexible Dataset**: Works with any CSV containing movie titles and plot descriptions

## Testing

Run the unit tests to ensure everything is working correctly:

```bash
# Run specific test file
uv run -m pytest test/test_movie_search.py

```

