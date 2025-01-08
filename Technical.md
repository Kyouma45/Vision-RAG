# VisRAG: Vision-Retrieval-Augmented Generation Application

VisRAG is a cutting-edge Vision-RAG (Retrieval-Augmented Generation) application tailored to process and extract relevant information from diverse input files, irrespective of their structure. By leveraging advanced Large Language Models (LLMs) and sophisticated architectures, VisRAG ensures exceptional accuracy, cost efficiency, and reduced inference time. Key technologies such as hybrid search, Optical Character Recognition (OCR), reranking, prompt engineering, and hyperparameter tuning minimize hallucination and maximize performance.

---

## Key Features

### 1. **Multi-format Data Parsing**
- **PDF Parsing**: Converts each PDF page into grayscale images and processes them in batches, optimizing for cost efficiency without compromising accuracy.
- **XML Parsing**: Users can upload XML files independently or alongside PDFs. Extracted data from XMLs is seamlessly merged with PDF data to enrich the knowledge database.
- **Image Formats**: Supports JPEG, JPG, and PNG files for versatile data input.
- **TXT Parsing**: Users can upload TXT files independently or alongside PDFs. Extracted data from TXTs is seamlessly merged with PDF data to enrich the knowledge database.

### 2. **Efficient Batch Processing**
- **OCR Capabilities**: Powered by OpenAI’s 4o model for precise text extraction.
- **Cost Savings**: Batches reduce processing costs by 50%.
- **Timeframe**: While batch processing may take up to 24 hours, real-time processing is available for urgent requests.
- **Real-time Status Monitoring**: The **Check Project Status** feature provides instant updates on pending and completed projects, ensuring transparency.

### 3. **Advanced Data Embeddings**
- **Embedding Model**: OpenAI’s embedding model (model details can be specified) generates high-dimensional (728-dimension) embeddings for superior data representation.
- **Storage Solution**:
  - Embeddings stored in a serverless database powered by Pinecone for robust similarity search.
  - Text stored in:
    - Embedding format for semantic search.
    - Count vectors for BM25 keyword-based search.
- **Hybrid Search**: Combines semantic embeddings with BM25 for unparalleled query precision.
- **Accuracy Enhancements**:
  - Hyperparameter tuning and threshold optimization reduce hallucinations.
  - Cohere’s reranker improves result accuracy by an additional 2%.

### 4. **KPI Management**
- **KPI Upload**: Users can upload peer benchmarking templates in CSV format to streamline KPI extraction.
- **Automated Reports**:
  - Generates peer benchmarking and KPI responses automatically.
  - Reports available for download in **Excel** and **CSV** formats.
- **Manual Reports**: Generate ad-hoc responses with the **Generate Responses** button for flexibility.

### 5. **Interactive Querying**
- Integrated **chat box** for querying project-specific questions and retrieving detailed insights interactively.

### 6. **Multiple Aliases**
- Supports up to 5 aliases for each KPI (e.g., GHG, Greenhouse Gas, Emission) to accommodate variations in terminology across documents.

### 7. **Document Translation**
- Users can specify a target output language for document translation.
- Extracted text is translated into the specified language and can be downloaded in **TXT** format.
- The chat interface also supports interaction in the target language, allowing users to query and engage in the specified language seamlessly.

### 8. **Synchronisation**
- 

---

## Workflow

### **Document Processing**
1. **Upload Documents**: Upload PDFs, XMLs, images, or text files via the user-friendly interface.
2. **Data Extraction**:
   - PDF pages are converted into grayscale images for OCR processing.
   - XML and TXT files are parsed directly using structured extraction techniques.
3. **Batching**: Documents are processed in batches to optimize cost and time. Users can opt for real-time processing if needed.
4. **Translation**: Checks for language compliance and translates extracted text to the desired output language if necessary.
5. **Status Monitoring**: Use the **Check Project Status** feature to track progress in real-time.
6. **Data Merging**: Combines processed data with supplementary sources (XML, TXT, etc.) to create a comprehensive knowledge base.
7. **Space Compression**: Reduces the size of processed projects by cleaning temporary data and compressing results.

### **Embedding Creation**
1. Converts extracted text into high-dimensional embeddings using OpenAI’s model.
2. Embeddings are stored in Pinecone for efficient retrieval and hybrid search.
3. Enhanced with hyperparameter tuning and reranking for improved query accuracy.

### **KPI Handling**
1. Upload peer benchmarking templates in CSV format via the **Update KPIs** feature.
2. Automatically extracts and stores KPI data for backend benchmarking.
3. Generate detailed reports or query insights through the chat box.

---

## Usage Instructions

1. **Upload Documents**:
   - Supported formats: PDFs, JPEG, JPG, PNG, XML, TXT.
   - Use the upload interface for seamless file submission.
2. **Monitor Project Status**:
   - Track progress via the **Check Project Status** feature.
3. **Generate Reports**:
   - Pre-generated reports can be downloaded, or manual responses can be generated with the **Generate Responses** button.
4. **Translate Documents**:
   - Use the **Translate TXT** button to download translated text files.
   - Specify a target language to translate input files and interact in the desired language through the chat interface.
5. **Query Projects**:
   - Interact with the chat box for project-specific questions or insights.

---

## Output Formats

- **Reports**:
  - Peer benchmarking and KPI responses in **Excel** and **CSV** formats.
- **Translated Documents**:
  - Text output available in **TXT** format.
- **Search Results**:
  - Delivered through hybrid search mechanisms combining embeddings and BM25.

---

## Benefits

- **Cost Efficiency**: Batch processing reduces OCR costs by up to 50%.
- **Scalability**: Seamlessly handles structured and unstructured data.
- **High Accuracy**: Leverages state-of-the-art embedding, reranking, and hyperparameter tuning techniques.
- **User-Friendly**: Intuitive interfaces for uploads, status checks, and report generation.
- **Language Support**: Specify a target output language for translations and chat interactions, enhancing accessibility and flexibility.
- **Customizability**: Alias management and KPI updates allow tailored experiences for diverse use cases.

---



## Code Walkthrough

### **main.py**
This central script connects all custom modules and orchestrates the workflow.

---

### **pdf.py**
Handles the parsing of PDF and XML documents, creating batches for the LLM to process or storing results for later use. It supports multiple file uploads (PDFs, XMLs, or both).

#### Functions:
1. **parse_files**:  
   - Accepts inputs along with metadata (e.g., project name, sector).  
   - Updates active projects (`sectors.json`).  
   - Delegates processing to `parse_pdfs` or `parse_xml`.

2. **parse_pdfs**:  
   - Converts PDF to grayscale images.  
   - Creates LLM-compatible batches for processing.  
   - Tracks batches and returns metadata upon completion.

3. **parse_xml**:  
   - Extracts text and metadata from XML files.

4. **save_sectors**:  
   - Updates the `sectors.json` file with new data.

5. **retry_request_create**:  
   - Implements a retry mechanism for LLM batch requests.

6. **Other utilities**:  
   - `retry_request`, `xml_to_dict`, `load_sectors`, `Encode_image`.

---

### **status.py**
Manages project and batch statuses.

#### Functions:
- **retry_request**: Implements retry logic for request failures.  
- **Check_project_batch_status**: Monitors batch processing progress.

---

### **vectorstore.py**
Handles vector storage and metadata preparation.

#### Functions:
1. **prepare_metadata**: Extracts metadata from Langchain documents and converts them into the format Pinecone requires. 
2. **validate_index_name**: Converts the project name into the format Pinecone requires.
3. **create_vectorstore**: Creates a serverless vector store using Pinecone.

---

### **Utilities.py**
It contains helper functions for managing data, prompts, and sectors.

#### Functions:
1. **File Management**:
   - `load_prompts`, `load_sectors`, `load_data`, `save_data`, `save_sectors`.

2. **Data Handling**:
   - `to_excel`, `extract_text_from_jsonl`, `compress_parsed_pdfs`.

3. **Project and Sector Management**:
   - `validate_index_name`, `refresh_data`, `update_KPIs`.  
   - `create_sector`, `create_project`, `load_project`, `delete_project`, `delete_sector`.

4. **Interactive Tools**:
   - `chat_interface`, `project_status`.

---

## Features
1. **Multi-File Support**: Simultaneous handling of PDFs and XMLs.
2. **Batch Processing**: Efficiently creates LLM batches for large datasets.
3. **Metadata Management**: Dynamically updates and stores project and sector data.
4. **Retry Mechanism**: Ensures robust request handling with retries.
5. **Vector Storage**: Prepares, validates, and creates LLM-compatible vector stores.

---

## Usage
1. **Prepare Inputs**: Organize PDFs/XMLs and required metadata.
2. **Run main.py**: Integrates modules and initiates the workflow.
3. **Monitor Progress**: Use `status.py` to check batch and project statuses.
4. **Manage Sectors and Projects**: Utilities in `Utilities.py` support adding, deleting, and refreshing data.

---

## Future Extensions
- Support for additional file formats.
- Enhanced retry logic with exponential backoff.
- Improved visualization of processing statuses.

---

VisRAG is your ultimate solution for efficient, accurate, and scalable document processing and information retrieval. Designed for modern needs, it’s an essential tool for any organization aiming to harness the power of AI-driven data processing and analysis.

