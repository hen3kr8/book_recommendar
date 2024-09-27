from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

app = Flask(__name__)

# Load the embedding model and the FAISS index at the start to avoid reloading it with each request

embeddedder_miniLM_6 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
new_vector_store = FAISS.load_local('embeddings/docsearch_mini', embeddings=embeddedder_miniLM_6, allow_dangerous_deserialization=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['book_name']
        results = inference_on_loaded_embedding(query)
        return render_template('index.html', results=results)
    return render_template('index.html')

def inference_on_loaded_embedding(query):
    # Perform the search on the loaded embeddings
    ans_mini = new_vector_store.similarity_search(query)
    return [result.page_content for result in ans_mini]




if __name__ == "__main__":
    app.run(debug=True)
