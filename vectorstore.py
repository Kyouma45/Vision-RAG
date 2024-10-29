import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
import time

load_dotenv()


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()


@st.cache_resource
def get_pinecone_client():
    return Pinecone()


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

    # Ensure name isn't too long (Pinecone has a 45 character limit)
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
    index_name = validate_index_name(index_name)
    st.info(f"Using index name: {index_name}")

    total_docs = len(docs)
    embeddings = get_embeddings()

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
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )
            status_text.text(f"Created new Pinecone index: {index_name}")
            # Wait for index to be ready
            time.sleep(10)

        # Initialize index with hybrid search
        vectorstore = pc.Index(name=index_name)
        vectorstore.describe_index_stats()  # Verify index is accessible

    except Exception as e:
        st.error(f"Error creating/checking Pinecone index: {e}")
        st.error(f"Details: {str(e)}")
        return None

    # Add documents to the index in batches
    status_text.text("Adding documents to Pinecone...")
    try:
        embeddings_model = get_embeddings()

        for i in range(0, total_docs, batch_size):
            batch = docs[i:i + batch_size]
            batch_embeddings = embeddings_model.embed_documents([doc.page_content for doc in batch])

            # Prepare vectors for upsert
            vectors = []
            for j, doc in enumerate(batch):
                metadata = prepare_metadata(doc)
                vectors.append({
                    "id": f"doc_{i + j}",
                    "values": batch_embeddings[j],
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

    st.success("Vector store created successfully with hybrid search enabled!")

    return vectorstore


def hybrid_search(
        vectorstore: Pinecone,
        query: str,
        k: int = 4,
        alpha: float = 0.5,
        filter: Dict[str, Any] = None
) -> List[Document]:
    """
    Perform hybrid search on the Pinecone index with metadata filtering

    Args:
        vectorstore: Pinecone vector store instance
        query: Search query string
        k: Number of results to return
        alpha: Balance between sparse and dense retrieval (0.0-1.0)
        filter: Metadata filter conditions

    Returns:
        List of relevant documents with scores
    """
    try:
        embeddings = get_embeddings()
        query_embedding = embeddings.embed_query(query)

        results = vectorstore.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            include_values=False,
            filter=filter
        )

        # Convert results to Documents
        documents = []
        for match in results.matches:
            doc = Document(
                page_content=match.metadata.get("text", ""),
                metadata={k: v for k, v in match.metadata.items() if k != "text"}
            )
            documents.append((doc, match.score))

        return documents
    except Exception as e:
        st.error(f"Error performing hybrid search: {e}")
        st.error(f"Details: {str(e)}")
        return []