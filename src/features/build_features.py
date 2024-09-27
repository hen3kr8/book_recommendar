# script to convert processed data into features for modeling

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import pandas as pd
from typing import List, Tuple
from langchain_core.documents import Document
from book_data import BookData
from flask import Flask, render_template, request
from collections import defaultdict
import re
from dotenv import load_dotenv

load_dotenv()
api_token = os.getenv('YOUR_API_TOKEN')
# Path processed data: RAG_book_repo/data/processed/book_genre.csv
USE_LOADED_EMBEDDINGS = True
app = Flask(__name__)

# Load the embedding model and the FAISS index at the start to avoid reloading it with each request
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

embeddedder_miniLM_6 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
new_vector_store = FAISS.load_local('embeddings/docsearch_mini', embeddings=embeddedder_miniLM_6, allow_dangerous_deserialization=True)
book_data_object = BookData('data/processed/book_genre.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_book_1_name = request.form['book_1_name']
        query_book_2_name = request.form['book_2_name']

        embedding_book_1 = get_embedding_for_book(query_book_1_name)
        embedding_book_2 = get_embedding_for_book(query_book_2_name)

        results_1 = inference_on_loaded_embedding(embedding_book_1)
        results_2 = inference_on_loaded_embedding(embedding_book_2)

        combined_results = combine_results([results_1, results_2])
        structured_results = []
        for result, score in combined_results:
            structured_results.append(parse_page_content(result))

        return render_template('index.html', results=structured_results)
    return render_template('index.html')


def get_embedding_for_book(query_book_name:str) -> List[float]:
    """
    Retrieves the embedding for a given book.
    Args:
    query_book_name (str): The name of the book to retrieve the embedding for.
    Returns:
    List[float]: The embedding of the book as a list of floats.
    """

    query_book = book_data_object.find_closest_book(query_book_name)
    book_document = convert_df_to_document(query_book)
    transformed_book = transform_data(book_document) # transform book into chunks, then into embedding.
    embedded_book = create_single_embedding(transformed_book)
    return embedded_book

def combine_results(embeddings: List[List[float]]) -> List[Tuple[List[float], float]]:
    """
    Combine the results from multiple embeddings into a single similarity score.
    Args:
        embeddings (List[List[float]]): A list of embeddings, where each embedding is a list of floats.
    Returns:
        List[Tuple[List[float], float]]: A list of tuples containing the document and its similarity score, sorted in descending order of score.
    """

    similarity_results = defaultdict(float)
    for embedding in embeddings:
        # Add the results to the similarity_results dict (count or rank the documents)
        for doc in embedding:
            similarity_results[doc] += 1 

    sorted_results = sorted(similarity_results.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_results

def transform_data(data) -> List[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts

def convert_df_to_document(df:pd.DataFrame) -> List[Document]:
    documents = []
    for _, row in df.iterrows():
        page_content = f"Book: {row['Book']}, Genres: {row['Genres']}, Description: {row['Description']}"
        document = Document(page_content=page_content)
        documents.append(document)
    return documents

def inference_on_loaded_embedding(embedding: List[float]) -> List[str]:
    """
    Perform inference on loaded embeddings. Texts are the embeddings of the book descriptions
    Args:
        embedding (List[float]): The embedding of the book description.
    Returns:
        List[str]: A list of page contents for the most similar embeddings.
    """
    
    new_vector_store = FAISS.load_local('embeddings/docsearch_mini', embeddings=embeddedder_miniLM_6, allow_dangerous_deserialization=True)
    ans_mini = new_vector_store.similarity_search_by_vector(embedding)
    
    return [result.page_content for result in ans_mini]


def parse_page_content(content: List[float]):
    """
    Parse the page_content string into a structured dictionary with book, genres, and description.
    """
    book_match = re.search(r"Book: (.*?)\n", content)
    genres_match = re.search(r"Genres: (.*?)\n", content)
    description_match = re.search(r"Description: (.*)", content)

    return {
        "book": book_match.group(1) if book_match else "Unknown",
        "genres": genres_match.group(1) if genres_match else "Unknown",
        "description": description_match.group(1) if description_match else "No description available"
    }


def create_and_save_embedding(texts):
    embeddedder_miniLM_6 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch_mini = FAISS.from_documents(texts, embeddedder_miniLM_6)
    docsearch_mini.save_local('embeddings/docsearch_mini')

def create_single_embedding(text:Document) -> List[float]:
    text = text[0]
    text_to_embed = text.page_content
    embedding = embeddedder_miniLM_6.embed_query(text_to_embed)  # Use embed_query to get the embedding for the text
    return embedding

def main():

    query_book_name = input('Enter a book to find books similar to it: ')
    if query_book_name is None:
        query_book_name = "To Kill A Mocking Bird"

    if USE_LOADED_EMBEDDINGS:
        book_data_object = BookData('data/processed/book_genre.csv')
        processed_data = book_data_object.book_data_documents
        results = inference_on_loaded_embedding(embedded_book)

    else:

        book_data_object = BookData('data/processed/book_genre.csv')
        processed_data = book_data_object.book_data_documents
        texts = transform_data(processed_data)
        query_book = book_data_object.find_closest_book(query_book_name)
        transformed_book = transform_data(query_book) # transform book into chunks, then into embedding.
        embedded_book = create_single_embedding(transformed_book)
        
   
if __name__ == "__main__":
    app.run(debug=True)

