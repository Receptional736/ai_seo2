from dotenv import load_dotenv


from typing import Dict,Literal
import os
import asyncio
from openai import OpenAI


import asyncio
from typing import List
from datetime import datetime
from pinecone import Pinecone

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(override=True)



# add docs, so it gets the docs and chunk it first and then save it there.


class PineconeDocStore:
    """Simple wrapper for embedding, upserting and querying text docs."""

    def __init__(
        self,

        dimension: int = 3072,
        index_name: str = "brand-recognition",
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        chunk_size =500,
        chunk_overlap = 100
    ) -> None:

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        # Clients
        self._oai = OpenAI(api_key=self.openai_api_key)
        
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        self._pc = Pinecone(api_key=self.pinecone_api_key)
        self._index = self._pc.Index(index_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )



    # ---------------------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        """Get OpenAI embedding for a piece of text."""
        res = self._oai.embeddings.create(
            model="text-embedding-3-large",
            input=[text],
        )
        return res.data[0].embedding
        

        # ────────────────────────────────────────────────────────────────────────
        # 5) Text splitter configuration
        # ────────────────────────────────────────────────────────────────────────
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    


    def upsert_all(self,namespace:str,subject:str,text_src, batch_size: int = 500) -> None:
        """Embed every document and upsert to Pinecone in batches."""
        to_upsert = []
        
        now = datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")


        chunks = self.text_splitter.split_text(text_src)

        
        for idx, chunk in enumerate(chunks):
            # Generate a unique ID per chunk
            chunk_id = f"{timestamp}-{idx}"
            emb = self._embed(chunk)

            # Attach metadata to recall origin if needed
            metadata = {
                
                "source_file": subject,
                "chunk_index": idx,
                "text_length": len(chunk),
                "text": chunk,
            }

            to_upsert.append(
                    {
                        "id": chunk_id,
                        "values": emb,
                        "metadata": metadata,
                    }
                )
        
        self._index.upsert(vectors=to_upsert, namespace=namespace)


    def search_query(self, query: str,namespace:str, top_k: int = 5) -> List[dict]:
        """Return Pinecone matches for a text query."""
        emb = self._embed(query)
        res = self._index.query(
            vector=emb,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            #filter=filter or {},
        )
        return res.matches
        
        