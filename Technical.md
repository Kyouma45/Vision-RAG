# Vision-RAG: Vision-Retrieval-Augmented Generation Application Documentation

Vision-RAG is an advanced application designed to process and extract relevant information from any type of PDF, regardless of its structure. It also parses infographics and accurately extracts data from formatted tables. By leveraging the power of Large Language Models (LLMs), Vision-RAG delivers exceptional accuracy and efficiency.

---

## Key Features

### 1. **Multi-format Data Parsing**
- **PDF Parsing**: Vision-RAG converts each PDF page into grayscale images and processes them in batches for cost-efficiency.
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

Vision-RAG provides a robust, efficient, and accurate solution for parsing and analyzing complex documents, making it an essential tool for data-driven decision-making.
