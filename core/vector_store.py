import time
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Union

from pinecone import Pinecone, ServerlessSpec
from config.settings import settings


class PineconeVectorStore:
    """
    Pinecone-based vector store with:
    - Managed Serverless infrastructure
    - Native Cloud-size Metadata Filtering (Pre-filtering)
    - Persistence (Cloud)
    """
    
    def __init__(self, dimension:int, index_name:str = "my-app-index" ):
        self.dimension = dimension
        self.index_name = index_name
        # Initialize Pinecone Client
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Check if index exists, create if not 
        # Note: In production, you might create the index manually in the console
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            self.pc.create_index(
                name= index_name,
                dimension= dimension,
                metric= "cosine",       # or 'dot-product' to mathc 'ip'
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
                
        self.index = self.pc.Index(index_name)
        
        # We only need to track next_id locally if you want sequential integer IDs.
        # Pinecone stores the data, so we don't store the index blob locally.
        self.next_id = 0
        
    
    def add_document(self, embeddings: np.ndarray, metadatas: List[Dict]) -> List[int]:
        """
        Upsert documents with embeddings and metadata to Pinecone
        """
        num_elements = len(embeddings)
        # Generate IDs (Pinecone requires IDs to be strings)
        ids = list(range(self.next_id, self.next_id + num_elements))
        str_ids = [str(_id) for _id in ids]
        
        # Prepare batch for Pinecone (format: (id, vector, metadata))
        vectors_to_upserts = []
        for _id, vec, meta in zip(str_ids, embeddings, metadatas):
            # Ensure vector is a list, not numpy array
            vectors_to_upserts.append((_id, vec.tolist(), meta))
            
        # Batch upsert (Pinecone recommends batches of ~100-200 depending on size)
        batch_size = 100
        for i in range(0, len(vectors_to_upserts), batch_size): # (From, To, STEP)
            batch = vectors_to_upserts[i : i + batch_size]
            self.index.upsert(vectors=batch)
            
        self.next_id += num_elements
        return ids
    
    
    def search(
        self,
        query_embedding: np.ndarray,
        k:int = 100,
        filter_dict: Optional[Dict] = None
    ) -> Tuple[List[int]] | List[float] :
        """
        Search with Native Pinecone Metadata Pre-filtering.
        
        Args:
            query_embedding: Query vector
            K: Number of results
            filter_dict : Dictionary using Pinecone filter syntax.
                          Example: {"category": {"$eq": "news"}}
                          pass None for no filtering.
        """
        
        # Ensure query is a list
        vector_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
         
        # Pinecone native query supports 'filter' directly (Pre-filtering)
        response= self.index.query(
            vector=vector_list,
            top_k=k,
            include_metadata=True,
            filter=filter_dict  # <--- This performs true Pre-filtering on the server
        )
        
        # Parse resposne 
        # Pinecone return matches object
        ids = []
        scores =  []
        
        for match in response['matches']:
            # Convert ID back to int to match your application logic
            ids.append(int(match['id']))
            scores.append(match['score'])
            
        return ids, scores
    
    def save(self, path:str):
        """
        Sync local ID counter state.
        (The actual index saved in the cloud automatically).
        """
        
        os.makedirs(path, exist_ok=True)
        # We only save the ID counter so we don't overwrite IDs on restart
        with open(os.path.join(path, "local_state.json"), 'w') as f:
            json.dump({
                "next_id": self.next_id,
                # We can verify dimension on load, but data is in cloud
                "dimension": self.dimension, 
                "index_name": self.index_name
            }, f)
        
    
    def load(self, path:str):
        """
        Load local ID counter state.
        """
        state_file = os.path.join(path, "local_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                data = json.load(f)
                self.next_id = data.get("next_id", 0)
        
        # We don't "load" the index; we just ensure we are connected
        # which is handled in __init__