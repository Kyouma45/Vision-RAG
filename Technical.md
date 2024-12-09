# EnvintAI: Vision-RAG Application Documentation

EnvintAI is an advanced Vision-RAG (Retrieval-Augmented Generation) application designed to process and extract relevant information from any type of PDF, regardless of its structure. It also parses infographics and accurately extracts data from formatted tables. By leveraging the power of Large Language Models (LLMs), EnvintAI delivers exceptional accuracy and efficiency. It uses advanced architecture like hybrid search, OCR, Reranker, Prompt Engineering, Hyperparameter Tunning, etc to reduce hallucination, cost, and inferance time while increasing accuracy.

---

## Key Features

### 1. **Multi-format Data Parsing**
- **PDF Parsing**: EnvintAI converts each PDF page into grayscale images and processes them in batches for cost-efficiency.
- **XML Parsing**: Users can upload XML files either independently or alongside PDFs. The extracted data from XMLs is merged with PDFs to enhance the knowledge database.

### 2. **Efficient Batch Processing**
- Utilizes OpenAI’s **4o model** for OCR capabilities.
- Batches reduce processing costs by 50%, although results may take up to 24 hours.
- The **Check Project Status** feature provides a real-time list of pending projects and updates project completion status automatically.

### 3. **Advanced Data Embeddings**
- **Embedding Model**: OpenAI’s embedding model (insert model name) converts text into **728-dimension embeddings**.
- **Storage**: 
  - Embeddings are stored in a serverless database powered by Pinecone.
  - Text is stored in two formats:
    - Embeddings for similarity search.
    - Count vectors for BM25 search.
- **Hybrid Search**: Combines similarity and BM25 searches for the most accurate query results.
- **Enhanced Accuracy**: Hyperparameter tuning filters and threshold values prevent model hallucination, while Cohere’s reranker improves accuracy by 2%.

### 4. **KPI Management**
- **Update KPIs**: Allows users to upload peer benchmarking templates in CSV format. The application automatically extracts and stores KPI information in the backend.
- **Report Generation**:
  - Automatically generates responses to KPIs for any completed project.
  - Peer benchmarking reports are downloadable in **Excel** and **CSV** formats.
  - A **Generate Responses** button enables manual report generation.

### 5. **Interactive Querying**
- Includes a **chat box** for querying any project-related questions.

### 6. **Multiple Alias**
- You can provide up to 5 different aliases for one single KPI. For eg: GHG, Greenhouse Gas, Emission, etc. This is done because same word can be written differently in all the PDFs depending on the company.

---

## Workflow

### **PDF Processing**
1. **Upload PDFs**: Extracted into grayscale images for processing.
2. **Batching**: Batches are sent to OpenAI’s 4o model for OCR.
3. **Status Check**: Use the **Check Project Status** feature to monitor progress.
4. **Completion**: Processed data is merged with XML data (if provided).

### **Embedding Creation**
1. Text converted into embeddings using OpenAI’s model.
2. Stored in Pinecone for hybrid search capabilities.
3. Enhanced with hyperparameter tuning and reranking.

### **KPI Handling**
1. Upload CSV templates via the **Update KPIs** feature.
2. Extracted KPI data is stored in the backend for benchmarking.
3. Generate reports or use the chat box for detailed insights.

---

## Usage Instructions

1. **Upload Documents**: 
   - Upload PDFs and/or XML files via the interface.
2. **Monitor Project Status**: 
   - Check pending and completed projects through **Check Project Status**.
3. **Generate Reports**:
   - Use the **Generate Responses** button or download pre-generated reports.
4. **Query Projects**:
   - Use the chat box for specific project-related questions.

---

## Output Formats

- Peer benchmarking reports: **Excel** and **CSV**
- Search results: Based on hybrid search using embeddings and BM25.

---

## Benefits

- **Cost-Efficient**: Reduced OCR processing costs through batching.
- **Scalable**: Handles both structured and unstructured data seamlessly.
- **Accurate**: Combines state-of-the-art embedding and ranking techniques for superior results.
- **User-Friendly**: Interactive status checks, intuitive report generation, and a query chat box for convenience.

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

EnvintAI provides a robust, efficient, and accurate solution for parsing and analyzing complex documents, making it an essential tool for data-driven decision-making.
