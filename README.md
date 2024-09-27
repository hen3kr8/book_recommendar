# README

## Overview

This application is a Flask-based web app that allows users to compare books based on their embeddings. It utilizes machine learning models and a pre-built FAISS index to perform similarity search on book descriptions. The app loads pre-embedded vectors and allows users to input two book names. The application then computes the embeddings of the books and retrieves similar books from the stored embeddings for comparison.

### Key Components:
- **Flask Web App**: Provides a web interface to input book names and receive results.
- **FAISS Vector Search**: A pre-built FAISS index is used to perform similarity search on book embeddings.
- **HuggingFace Embeddings**: HuggingFace's `all-MiniLM-L6-v2` embedding model is used to generate vector embeddings for the book descriptions.
- **Processed Data**: The data for the books is loaded from a CSV file (`data/processed/book_genre.csv`), which contains information about books, genres, and descriptions.
- **Embedding Storage**: The embeddings are stored locally in the `embeddings/docsearch_mini` FAISS index.

## How the Application Works

1. **Data Loading**: On app initialization, the processed book data is loaded from a CSV file. This data contains information about books such as their name, genre, and description.
2. **Embedding Search**: When the user submits two book names through the web interface, the app retrieves their embeddings using the pre-loaded FAISS index.
3. **Similarity Search**: The embeddings of the submitted books are used to perform a similarity search, retrieving similar book descriptions from the FAISS index.
4. **Results Display**: The app parses the retrieved book descriptions and displays them to the user in a structured format with information about the book's name, genres, and description.

## Requirements

To run this application, you need to install the necessary Python packages listed in `requirements.txt`. Make sure to install these dependencies in a virtual environment.

### Installation

1. **Clone the Repository**:
    ```bash
    git clone https://your-repo-url
    cd your-repo-url
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate    # For Linux/MacOS
    venv\Scripts\activate       # For Windows
    ```

3. **Install Dependencies**:
    Install the required dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Environment Variables**:
    Create a `.env` file in the root of the project and add your HuggingFace API token:
    ```bash
    YOUR_API_TOKEN=your_huggingface_api_token
    ```

## How to Run

### Flask Application

To start the Flask web application:

```bash
python src/features/build_features.py
```

This will run the Flask server locally. You can then access the web interface by navigating to `http://127.0.0.1:5000/` in your browser.

### Main Function (Feature Building)

To run the feature extraction and embedding generation script separately (without the web app):

```bash
python src/features/build_features.py
```

This script:
- Prompts the user to input a book name.
- Retrieves or computes the embedding for the input book.
- Performs similarity search on the stored embeddings to find similar books.

## Folder Structure

- **`data/processed/book_genre.csv`**: The CSV file containing processed book data with names, genres, and descriptions.
- **`embeddings/docsearch_mini`**: Local FAISS index storing precomputed book embeddings.
- **`src/features/build_features.py`**: The main script responsible for starting the Flask app and processing data.
- **`templates/index.html`**: HTML file for the web interface.

## Additional Features

- **Embedding Generation**: New embeddings for book data can be generated using the `HuggingFaceEmbeddings` model and stored locally in FAISS format.
- **Inference on Loaded Embeddings**: Inference is performed on preloaded embeddings to retrieve the most similar books.

## License

Specify the license for your application.

## Acknowledgments

This project uses:
- [LangChain](https://github.com/hwchase17/langchain)
- [HuggingFace](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)

