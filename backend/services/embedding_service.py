import chromadb
from chromadb.config import Settings
import openai
import google.generativeai as genai
from typing import List, Dict
import os

class EmbeddingService:
    def __init__(self):
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = os.getenv("CHROMA_PORT", "8000")
        
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=int(chroma_port)
        )
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    def get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def get_gemini_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Gemini"""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings
    
    def create_collection(self, collection_name: str):
        """Create or get a collection"""
        try:
            return self.chroma_client.create_collection(name=collection_name)
        except:
            return self.chroma_client.get_collection(name=collection_name)
    
    def add_documents(
        self, 
        collection_name: str, 
        chunks: List[Dict],
        embedding_model: str = "openai"
    ):
        """Add documents to ChromaDB"""
        collection = self.create_collection(collection_name)
        
        texts = [chunk["content"] for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]
        
        if embedding_model == "openai":
            embeddings = self.get_openai_embeddings(texts)
        else:
            embeddings = self.get_gemini_embeddings(texts)
        
        collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids
        )
        
        return {"status": "success", "count": len(chunks)}
    
    def query_collection(
        self, 
        collection_name: str, 
        query_text: str,
        embedding_model: str = "openai",
        n_results: int = 3
    ) -> List[Dict]:
        """Query the collection"""
        collection = self.chroma_client.get_collection(name=collection_name)
        
        if embedding_model == "openai":
            query_embedding = self.get_openai_embeddings([query_text])[0]
        else:
            query_embedding = self.get_gemini_embeddings([query_text])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return [{
            "content": doc,
            "distance": dist
        } for doc, dist in zip(results['documents'][0], results['distances'][0])]