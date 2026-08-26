# NovaTech Enterprise Knowledge Assistant with Advanced RAG

## 1. Project Overview

The NovaTech Enterprise Knowledge Assistant is a Retrieval-Augmented Generation (RAG) application designed to help employees quickly find reliable information from internal company documents.

The solution combines semantic vector search, BM25 keyword search, hybrid retrieval, reranking, conversational memory, and grounded LLM responses with source citations.

The application is built using Python, LangChain, OpenAI, ChromaDB, Sentence Transformers, and Streamlit.


Documents
  -> Loader
  -> Chunking
  -> Embeddings
  -> ChromaDB

User Query
  -> Query Rewrite
  -> Vector Search + BM25
  -> Hybrid Retrieval
  -> Reranking
  -> Final Context Selection
  -> LLM
  -> Answer + Sources

  doc/arch.png 


## 2. Problem Statement

Employees often need to search across multiple internal policy documents such as:

- Leave policies
- Travel policies
- IT security policies
- Employee benefits
- Remote work policies
- Code of conduct
- Employee handbook
- Company FAQs

Traditional keyword search may miss semantically relevant information, while a general-purpose LLM may generate unsupported answers.

The goal of this project is to build an enterprise knowledge assistant that:

- retrieves relevant information from company documents
- combines semantic and keyword retrieval
- reranks retrieved content for improved relevance
- supports conversational follow-up questions
- provides source references
- minimizes hallucination
- returns a clear fallback response when information is not available



## 3. Solution Overview

The application uses an advanced RAG pipeline.

### Offline Ingestion and Indexing

Company Documents
↓
Document Loader
↓
Text Chunking
↓
OpenAI Embeddings
↓
ChromaDB Vector Store

The ingestion process is executed separately using:

bash
python ingest.py
This allows document indexing to be separated from the user-facing application.
Online Query Flow
User Question
↓
Conversation-Aware Query Rewrite
↓
Vector Search + BM25 Search
↓
Hybrid Retrieval
↓
Cross-Encoder Reranking
↓
Final Context Selection
↓
LLM
↓
Answer + Source Citations

## 4. Key Features

Multi-Format Document Support
The application supports multiple document formats:
PDF
DOCX
TXT
Documents are loaded from the local data/ directory.
Document Chunking
Documents are split into smaller overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
Chunk size and overlap are configurable through config.yaml.
Semantic Vector Retrieval
OpenAI embeddings are generated for document chunks and persisted locally using ChromaDB.
Semantic similarity search helps retrieve information even when the user's wording differs from the source document.
BM25 Keyword Retrieval
BM25 is used as a keyword-based retrieval mechanism.
A normalized tokenizer is used to improve matching between phrases such as:
carry-forward
and:
carry forward
Hybrid Search
Vector search and BM25 results are combined using configurable weights.
Example configuration:
hybrid_search:
 vector_weight: 0.6
  keyword_weight: 0.4

This allows document indexing to be separated from the user-facing application.
Online Query Flow
User Question
↓
Conversation-Aware Query Rewrite
↓
Vector Search + BM25 Search
↓
Hybrid Retrieval
↓
Cross-Encoder Reranking
↓
Final Context Selection
↓
LLM
↓
Answer + Source Citations
4. Key Features
Multi-Format Document Support
The application supports multiple document formats:
PDF
DOCX
TXT
Documents are loaded from the local data/ directory.
Document Chunking
Documents are split into smaller overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
Chunk size and overlap are configurable through config.yaml.
Semantic Vector Retrieval
OpenAI embeddings are generated for document chunks and persisted locally using ChromaDB.
Semantic similarity search helps retrieve information even when the user's wording differs from the source document.
BM25 Keyword Retrieval
BM25 is used as a keyword-based retrieval mechanism.
A normalized tokenizer is used to improve matching between phrases such as:
carry-forward
and:
carry forward
Hybrid Search
Vector search and BM25 results are combined using configurable weights.
Example configuration:
hybrid_search:
  vector_weight: 0.6
  keyword_weight: 0.4


This combines the strengths of semantic retrieval and keyword matching.
Cross-Encoder Reranking
Retrieved candidates are reranked using:
cross-encoder/ms-marco-MiniLM-L-6-v2
The reranker evaluates the relevance of each retrieved chunk against the user query.
Final Candidate Selection
The final context preserves strong candidates from both:
hybrid retrieval
reranked retrieval
A final ranking score combines:
reranker score
hybrid retrieval score
source priority
This helps prevent a strong candidate identified by hybrid retrieval from being completely removed by the reranking stage.
Conversational Memory
The assistant maintains recent conversation history.
For follow-up questions such as:
What is the annual leave policy?
followed by:
What about carry-forward?
the system rewrites the follow-up question into a standalone retrieval query before searching the knowledge base.
Source Citations
The application displays the source documents used to generate the answer.
This improves transparency and allows users to verify information against the original company policies.
Hallucination Mitigation
The LLM is instructed to answer only from retrieved company context.
If the information is not available, the assistant returns a fallback response such as:
I could not find this information in the available company documents.
Example unsupported question:
What is NovaTech's company car allowance?
Streamlit Chat Interface
The application includes:
chat interface
conversation history
clear conversation button
expandable source list
loading spinner
user-friendly error messages
Technical exceptions are captured using application logging rather than exposed directly to the user.

## 5. Technology Stack

###     Component	      Technology
Programming Language	Python 3.11
RAG Framework	        LangChain
LLM	OpenAI
Embeddings	            OpenAI text-embedding-3-small
Vector Database	        ChromaDB
Keyword Search	        BM25
Reranking	Sentence Transformers       CrossEncoder
UI	Streamlit
Configuration	YAML
Environment Variables	python-dotenv
PDF Processing	pypdf
DOCX Processing	python-docx

### 6. Project Structure

enterprise-knowledge-assistant-streamlit/
│
├── app.py
├── ingest.py
├── config.py
├── config.yaml
├── requirements.txt
├── README.md
├── .env
├── .env.example
├── .gitignore
│
├── data/
│   ├── leave_policy.pdf
│   ├── employee_handbook.pdf
│   ├── travel_policy.pdf
│   ├── it_security_policy.pdf
│   ├── employee_benefits.docx
│   ├── remote_work_policy.docx
│   ├── code_of_conduct.docx
│   └── company_faq.txt
│
├── chroma_db/
│
└── src/
    ├── document_loader.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    ├── vector_retriever.py
    ├── bm25_retriever.py
    ├── hybrid_retriever.py
    ├── reranker.py
    ├── rag_chain.py
    ├── memory.py
    ├── prompts.py
    └── logger.py

### 7. Setup Instructions

#### Prerequisites

Python 3.11
OpenAI API key
Step 1: Clone the Repository
git clone <repository-url>
cd enterprise-knowledge-assistant-streamlit
Step 2: Create a Virtual Environment
python3.11 -m venv .venv
Activate it:
source .venv/bin/activate
For Windows:
.venv\Scripts\activate
Step 3: Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

## 8. Environment Configuration

Create a .env file in the project root:
OPENAI_API_KEY=your_actual_openai_api_key
An example file is provided as:
.env.example
The real .env file must not be committed to Git.

#### 9. Application Configuration

Most non-secret settings are maintained in:
config.yaml
This includes:
LLM model
embedding model
chunk size
chunk overlap
retrieval counts
hybrid weights
reranker settings
memory size
source priority
final ranking weights
logging level
This avoids unnecessary hard-coded configuration in the application code.

### 10. Build or Refresh the Knowledge Index

Run:
python ingest.py
This executes:
Load Documents
→ Chunk Documents
→ Generate Embeddings
→ Persist Chunks in ChromaDB
Run ingestion again when:
documents are added
documents are updated
documents are removed
chunking configuration changes
embedding configuration changes

### 11. Run the Application

Start Streamlit:
streamlit run app.py
Streamlit will display a local URL such as:
http://localhost:8501
Open the URL in a browser.

### 12. Sample Questions

Leave Policy
What is the annual leave policy?
What about carry-forward?
When should I use the carried-forward leave?
Travel
What is the domestic hotel reimbursement limit?
When is premium economy allowed?
Employee Benefits
How much is the annual wellness benefit?
What is the annual learning benefit?
Remote Work
What is the monthly internet reimbursement limit?
IT Security
Can I store confidential company data in my personal Google Drive?
What should I do if I lose my company laptop?
Unsupported Information
What is NovaTech's company car allowance?
Expected behavior:
The assistant should state that the information could not be found in the available company documents.

### 13. Example Conversational Flow

User:
What is the annual leave policy?
Assistant:
Provides the annual leave policy based on company documents.
User:
What about carry-forward?
Assistant:
Understands that the follow-up refers to annual leave and retrieves the carry-forward policy.
User:
When should I use the carried-forward leave?
Assistant:
Returns the applicable usage deadline from the retrieved policy.
This demonstrates conversational memory and query rewriting.

### 14. Design Decisions

Separate Ingestion from Application Serving
The indexing pipeline is separated from the Streamlit application.
python ingest.py
is responsible for preparing the vector index.
streamlit run app.py
is responsible for serving user queries.
This avoids regenerating document embeddings whenever the user-facing application starts.
Hybrid Retrieval
Semantic search alone may miss exact policy terms.
BM25 alone may miss semantically related queries.
Using both provides better retrieval coverage.
Reranking After Retrieval
The cross-encoder evaluates retrieved chunks more precisely than the first-stage retrievers.
This improves the relevance of the final context provided to the LLM.
Candidate Preservation
A strong hybrid retrieval result may sometimes be scored lower by the reranker.
The final selection therefore preserves top candidates from both retrieval stages before calculating a final ranking.
Local Vector Storage
ChromaDB is used as a locally persisted vector database.
This provides simple local development and avoids requiring an external vector database service.
Configuration-Driven Design
Application settings are stored in config.yaml.
Sensitive information remains in .env.
This improves maintainability and prevents secrets from being hard-coded.

### 15. Logging and Error Handling

The application uses Python logging instead of operational print() statements.
Logging levels include:
INFO
DEBUG
WARNING
ERROR
EXCEPTION
User-facing Streamlit errors remain generic, while full technical exceptions are recorded in application logs.
Example log:
INFO | src.hybrid_retriever | Hybrid retrieval completed
INFO | src.reranker | Reranking completed
INFO | src.rag_chain | Final context selected
INFO | src.rag_chain | Query processed successfully

### 16. Current Limitations

Local Document Source
The current solution ingests files from a local directory.
It does not currently connect directly to enterprise content platforms such as SharePoint, Confluence, or Google Drive.
BM25 Rebuilt at Application Startup
The Chroma vector index is persisted locally.
However, the BM25 keyword index is currently rebuilt in memory from the source documents when the application initializes.
A future production implementation could persist or independently manage the keyword index.
Local Conversational Memory
Conversation history is stored in the Streamlit session.
It is not persisted across browser sessions or application restarts.
No Enterprise Authentication
The current project does not implement enterprise authentication, authorization, or document-level access control.
Local Development Architecture
The application is designed as a certification/demo implementation rather than a distributed production platform.
A production system may use:
managed vector databases
persistent chat storage
centralized observability
access control
container deployment
document synchronization
automated indexing pipelines

### 17. Future Enhancements

Possible future improvements include:
SharePoint or Confluence connectors
automated document ingestion
incremental index updates
persistent BM25 index
user authentication and authorization
role-based access control
document-level permissions
persistent conversation memory
retrieval evaluation metrics
automated RAG quality testing
monitoring and observability
containerized deployment
managed vector database
optional tool/function calling for enterprise workflows

### 18. Security Considerations

The OpenAI API key is stored using environment variables.
The .gitignore file prevents the following from being committed:
.env
.venv/
.venv_test/
__pycache__/
*.pyc
.DS_Store
chroma_db/
API keys and other secrets must never be committed to GitHub.

### 19. Demonstrated RAG Capabilities

This project demonstrates:
multi-format ingestion
chunking
embeddings
persistent vector storage
semantic retrieval
BM25 retrieval
hybrid search
cross-encoder reranking
conversational query rewriting
short-term conversational memory
context-based answer generation
source citations
hallucination mitigation
fallback handling
configuration management
structured logging
Streamlit user interface

## 20. Conclusion

The NovaTech Enterprise Knowledge Assistant demonstrates an advanced enterprise RAG architecture that combines multiple retrieval strategies, reranking, conversational memory, and grounded LLM generation.
The solution is designed to provide accurate, transparent, and maintainable access to internal company knowledge while reducing the risk of unsupported LLM-generated responses.
