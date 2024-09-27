import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from typing import List
from fuzzywuzzy import process

class BookData:

    def __init__(self, path_to_book_csv:str) -> None:
        self.path_to_book_csv = path_to_book_csv
        self.book_data_df = self.load_data_as_df()
        self.book_data_documents = self.load_data_as_Document()

    def load_data_as_df(self) -> pd.DataFrame:
        #  We need the data in this format to compare the titles to the user queries. 
        book_data_df = pd.read_csv(self.path_to_book_csv, encoding='utf-8')
        return book_data_df

    def load_data_as_Document(self) -> List[Document]:
        # we need the data in this format to evt put it in the embedder
        book_data_docs = CSVLoader(self.path_to_book_csv, encoding='utf-8')
        book_data_docs = book_data_docs.load()
        return book_data_docs
    
  
    def find_closest_book(self, book_name: str, match_type='fuzzy') -> pd.DataFrame:
        """
        Given the users query, which should be a book name, we use that to find the
        book in the dataframe with the closest title
        Args:
            book_name (str): Book name which the user entered
            match_type (str, optional): Type of match to perform. Defaults to 'fuzzy'.
        Returns:
            pd.DataFrame: The book found in the database with the closest title
        """        
        if match_type == 'fuzzy':
            all_book_titles = self.book_data_df['Book'].tolist()
            nearest_match = process.extractOne(book_name, all_book_titles)[0] # a tuple is returned (title, score). We only want the title
            nearest_row = self.book_data_df[self.book_data_df['Book'] == nearest_match]
            return nearest_row

        exact_match = self.book_data_df[self.book_data_df['Book'] == book_name]
        return exact_match
    
    
