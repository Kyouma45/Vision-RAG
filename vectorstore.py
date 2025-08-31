import streamlit as st
import re
from dotenv import load_dotenv
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder
from typing import List, Dict, Any, Tuple, Optional
# from langchain_core.retrievers import BaseRetriever
# from pydantic import BaseModel, Field
# from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import time

load_dotenv()


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()


@st.cache_resource
def get_pinecone_client():
    return Pinecone()


@st.cache_resource
def get_bm25_encoder():
    return BM25Encoder().default()


def prepare_metadata(doc: Document) -> Dict[str, Any]:
    """
    Prepare metadata for Pinecone document storage

    Args:
        doc: Document object with metadata

    Returns:
        Dictionary with processed metadata
    """
    metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}

    # Ensure text field is present for hybrid search
    metadata['text'] = doc.page_content

    # Add source if available
    if 'source' not in metadata and hasattr(doc, 'source'):
        metadata['source'] = doc.metadata['source']

    # Add page_number if available
    if 'page_number' not in metadata and hasattr(doc, 'page_number'):
        metadata['page_number'] = doc.metadata['page_number']

    return metadata


def validate_index_name(index_name: str) -> str:
    """
    Validate and format index name according to Pinecone requirements

    Args:
        index_name: Original index name

    Returns:
        Formatted valid index name
    """
    # Convert to lowercase
    index_name = index_name.lower()

    # Replace invalid characters with hyphens
    index_name = re.sub(r'[^a-z0-9-]', '-', index_name)

    # Remove leading/trailing hyphens
    index_name = index_name.strip('-')

    # Ensure name isn't too long (Pinecone has a 45-character limit)
    if len(index_name) > 45:
        index_name = index_name[:45].rstrip('-')

    # Ensure name isn't empty
    if not index_name:
        index_name = 'default-index'

    return index_name


def create_vectorstore(
        docs: List[Document],
        index_name: str,
        delete_existing: bool = True,
        batch_size: int = 100
):
    """
    Create a Pinecone vector store with hybrid search and metadata support

    Args:
        docs: List of Document objects
        index_name: Name of the Pinecone index
        delete_existing: Whether to delete existing index
        batch_size: Number of documents to process in each batch
    """
    # Validate and format index name
    path = index_name
    index_name = validate_index_name(index_name)
    st.info(f"Using index name: {index_name}")

    total_docs = len(docs)
    embeddings = get_embeddings()
    bm25_encoder = get_bm25_encoder()

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize Pinecone client
    try:
        pc = get_pinecone_client()
        status_text.text("Connected to Pinecone successfully")
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {e}")
        return None

    # Check if index exists, if not create it
    status_text.text("Checking Pinecone index...")
    try:
        indexes = pc.list_indexes()
        if index_name in indexes:
            if delete_existing:
                status_text.text(f"Deleting existing index: {index_name}")
                pc.delete_index(index_name)
                # Wait for deletion to complete
                time.sleep(5)
            else:
                st.warning(f"Index {index_name} already exists. Using existing index.")

        if index_name not in pc.list_indexes():
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )
            status_text.text(f"Created new Pinecone index: {index_name}")

        # Initialize index with hybrid search
        vectorstore = pc.Index(name=index_name)
        vectorstore.describe_index_stats()  # Verify index is accessible

    except Exception as e:
        st.error(f"Error creating/checking Pinecone index: {e}")
        st.error(f"Details: {str(e)}")
        return None

    # Prepare documents for BM25 encoding
    texts = [doc.page_content for doc in docs]

    # Fit BM25 encoder on all texts
    bm25_encoder.fit(texts)

    # Add documents to the index in batches
    status_text.text("Adding documents to Pinecone...")
    try:
        for i in range(0, total_docs, batch_size):
            batch = docs[i:i + batch_size]
            batch_texts = [doc.page_content for doc in batch]

            # Get dense embeddings for batch
            dense_embeddings = embeddings.embed_documents(batch_texts)

            # Get sparse embeddings for batch
            sparse_vectors = bm25_encoder.encode_documents(batch_texts)

            # Prepare vectors for upsert
            vectors = []
            for j, doc in enumerate(batch):
                metadata = prepare_metadata(doc)
                sparse_dict = sparse_vectors[j]

                vectors.append({
                    "id": f"doc_{i + j}",
                    "values": dense_embeddings[j],
                    "sparse_values": sparse_dict,
                    "metadata": metadata
                })

            if vectors:
                vectorstore.upsert(vectors=vectors)

            # Update progress
            progress = min((i + batch_size) / total_docs, 1.0)
            progress_bar.progress(progress)

        status_text.text("Successfully added documents to Pinecone")

    except Exception as e:
        st.error(f"Error adding documents to Pinecone: {e}")
        st.error(f"Details: {str(e)}")
        return None

    st.write("Vector store created successfully with hybrid search enabled!")

    return vectorstore


# def hybrid_search(
#         vectorstore, query: str, k: int = 4, alpha: float = 0.5, filter: Dict[str, Any] = None
# ) -> List[Document]:
#     """
#     Perform hybrid search on the Pinecone index with metadata filtering
#
#     Args:
#         vectorstore: Pinecone vector store instance
#         query: Search query string
#         k: Number of results to return
#         alpha: Balance between sparse and dense retrieval (0.0-1.0)
#         filter: Metadata filter conditions
#
#     Returns:
#         List of relevant documents with scores
#     """
#     try:
#         # Get dense embedding for query
#         dense_vec = st.session_state.embeddings.embed_query(query)
#
#         # Get sparse embedding for query
#         bm25_encoder = BM25Encoder().default()
#         sparse_vec = bm25_encoder.encode_queries([query])[0]  # Returns dict in correct format
#
#         results = vectorstore.query(
#             vector=dense_vec,
#             sparse_vector=sparse_vec,
#             top_k=k,
#             include_metadata=True,
#             include_values=False,
#             alpha=alpha,
#             filter=filter
#         )
#
#         # Convert results to Documents
#         documents = []
#         for match in results.matches:
#             doc = Document(
#                 page_content=match.metadata.get("text", ""),
#                 metadata={k: v for k, v in match.metadata.items() if k != "text"}
#             )
#             documents.append((doc, match.score))
#
#         return documents
#
#     except Exception as e:
#         st.error(f"Error performing hybrid search: {e}")
#         st.error(f"Details: {str(e)}")
#         return []
#
#
# # Create a context variable for Streamlit session state
# session_state_var = contextvars.ContextVar('session_state', default=None)
#
#
# class HybridRetriever(BaseRetriever):
#     """Custom retriever that implements hybrid search using Pinecone"""
#     class Config:
#         """Configuration for this pydantic object."""
#         arbitrary_types_allowed = True
#
#     vectorstore: Any = Field(description="Pinecone vectorstore instance")
#     k: int = Field(default=4, description="Number of results to return")
#     alpha: float = Field(default=0.5, description="Balance between sparse and dense retrieval")
#     filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter conditions")
#
#     def get_relevant_documents(
#             self,
#             query: str,
#             *,
#             callbacks: Optional[CallbackManagerForRetrieverRun] = None,
#             **kwargs: Any,
#     ) -> List[Document]:
#         """Get documents relevant to a query."""
#         try:
#             # Get the session state from context
#             session_state = session_state_var.get()
#             if session_state is None:
#                 session_state = st.session_state
#                 session_state_var.set(session_state)
#
#             # Call the hybrid_search function and extract just the documents
#             results = hybrid_search(
#                 vectorstore=self.vectorstore,
#                 query=query,
#                 k=self.k,
#                 alpha=self.alpha,
#                 filter=self.filter
#             )
#             return results
#         except Exception as e:
#             st.error(f"Error in get_relevant_documents: {str(e)}")
#             return []
#
#     async def aget_relevant_documents(
#             self,
#             query: str,
#             *,
#             callbacks: Optional[CallbackManagerForRetrieverRun] = None,
#             **kwargs: Any,
#     ) -> List[Document]:
#         """Asynchronously get documents relevant to a query."""
#         # For now, just call the sync version
#         return self.get_relevant_documents(query, callbacks=callbacks, **kwargs)