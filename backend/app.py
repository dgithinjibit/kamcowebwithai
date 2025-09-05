import os
import asyncio
import nest_asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from dotenv import load_dotenv
import uvicorn
import time

# Apply nest_asyncio patch
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Kamwenja TTC Chatbot API",
    description="RAG-powered chatbot for Kamwenja Teachers' Training College",
    version="1.0.0"
)

# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate API keys
if not PINECONE_API_KEY:
    raise RuntimeError("Pinecone API key is missing. Please set it in your .env file.")
if not GOOGLE_API_KEY:
    raise RuntimeError("Google API key is missing. Please set it in your .env file.")

# Initialize Pinecone and embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "kamwenja-ttc"
dimension = 768  # For Google's embedding-001 model

# Check if index exists; if not, create it
existing_indexes = pc.list_indexes()
if index_name not in [index.name for index in existing_indexes.indexes]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=None  # Using legacy spec for compatibility
    )
    print(f"Created Pinecone index: {index_name}")
    time.sleep(30)  # Wait for index to be ready

pinecone_index = pc.Index(index_name)
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define system prompt template for Kamwenja TTC
system_prompt_template = """
Your name is Kamwenja TTC Assistant. You are a specialized AI assistant for Kamwenja Teachers' Training College. 
Answer questions using ONLY the information provided in the context below. Be friendly, professional, and helpful.

{context}

Instructions:
- Answer questions using ONLY the information above
- If the answer is not in the knowledge base, say: "I don't have specific information about that in my knowledge base. Please contact the admissions office for assistance."
- Be concise and accurate
- Format your response clearly with bullet points when appropriate
"""

class QuestionRequest(BaseModel):
    question: str

def split_docs(documents, chunk_size=1500, chunk_overlap=100):
    """Split documents into chunks for embedding"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def embed_batch_with_retry(embed_model, batch_contents, max_attempts=3):
    """Embed a batch of documents with retry logic"""
    for attempt in range(max_attempts):
        try:
            return embed_model.embed_documents(batch_contents)
        except Exception as e:
            print(f"Error while embedding batch on attempt {attempt+1}: {e}")
            time.sleep(10 * (attempt + 1))  # Exponential backoff
            if attempt == max_attempts - 1:
                raise

def concurrent_embed_documents(embed_model, documents, batch_size=100, max_workers=4):
    """Parallelize embedding calls"""
    import concurrent.futures
    from tqdm import tqdm
    
    all_embeddings = []
    all_contents = []
    futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_contents = [doc.page_content for doc in batch]
            futures.append((executor.submit(embed_batch_with_retry, embed_model, batch_contents), batch_contents))
        
        for future, contents in tqdm(futures, total=len(futures), desc="Embedding batches"):
            try:
                batch_embeddings = future.result()
                all_embeddings.extend(batch_embeddings)
                all_contents.extend(contents)
            except Exception as e:
                print(f"Error in embedding batch: {e}")
    
    return all_embeddings, all_contents

def batch_upsert(index, vectors, batch_size=100):
    """Batch upsert vectors to Pinecone with retry logic"""
    batches = [vectors[i:i+batch_size] for i in range(0, len(vectors), batch_size)]
    
    for batch_number, batch in enumerate(batches):
        for attempt in range(3):
            try:
                index.upsert(vectors=batch)
                break
            except Exception as e:
                print(f"Upsert error on batch {batch_number+1}, attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"Batch {batch_number+1} failed after 3 attempts.")
                    raise e

@app.post("/ingest")
async def ingest_documents():
    """Ingest documents into the vector database"""
    try:
        # This would normally process uploaded files
        # For now, we'll use a placeholder document
        # In production, you'd accept file uploads and process them
        
        # Example: Simulate document ingestion
        sample_docs = [
            {
                "page_content": """
                Kamwenja Teachers' Training College Information

                ## Fee Structure (per term)
                - First Year: KES 55,000
                - Second Year: KES 50,000
                - Third Year: KES 48,000
                - Boarding Fees (Optional): KES 15,000 per term
                - Examination Fees (Payable in 2nd term): KES 5,000
                - Payment Methods: Payments can be made via the official school bank account (KCB Bank, Kamwenja Branch, Account No: 123456789) or via M-Pesa Paybill (Business No: 987654, Account: Student's Admission Number).
                """,
                "metadata": {"source": "knowledge_base", "type": "fees"}
            },
            {
                "page_content": """
                ## Courses Offered
                1. Diploma in Primary Teacher Education (DPTE):
                   - Duration: 3 years
                   - Minimum Requirement: KCSE Mean Grade of C (Plain).
                2. Diploma in Early Childhood Teacher Education (DECTE):
                   - Duration: 3 years
                   - Minimum Requirement: KCSE Mean Grade of C (Plain).
                3. Upgrading from Certificate to Diploma (Primary):
                   - Duration: 1 year (4 school terms)
                   - Requirement: Valid P1 Certificate.
                """,
                "metadata": {"source": "knowledge_base", "type": "courses"}
            },
            {
                "page_content": """
                ## Admission Requirements
                - Original and 2 copies of KCSE Certificate/Result Slip.
                - Original and 2 copies of National ID Card or Birth Certificate.
                - 4 recent passport-size photographs.
                - Completed application form (downloadable from the college portal).
                - Application fee receipt of KES 1,000.
                """,
                "metadata": {"source": "knowledge_base", "type": "admission"}
            },
            {
                "page_content": """
                ## Campus Life & Facilities
                - Library: Open from 8 AM to 9 PM, Monday to Saturday.
                - ICT Lab: Fully equipped with modern computers and internet access.
                - Sports: Facilities for Football, Volleyball, and Athletics are available.
                - Clubs: Drama club, Christian Union, Debate club.
                """,
                "metadata": {"source": "knowledge_base", "type": "campus"}
            },
            {
                "page_content": """
                ## Location
                - Kamwenja TTC is located in Nyeri County, Kenya.
                """,
                "metadata": {"source": "knowledge_base", "type": "location"}
            },
            {
                "page_content": """
                ## Important Dates (Next Intake)
                - Application Deadline: July 31st
                - Admission Letters Sent Out: August 15th
                - Reporting Date for New Students: September 5th
                """,
                "metadata": {"source": "knowledge_base", "type": "dates"}
            }
        ]
        
        # Split documents
        docs = []
        from langchain.docstore.document import Document
        
        for doc in sample_docs:
            docs.append(Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            ))
        
        split_docs_list = split_docs(docs, chunk_size=1000, chunk_overlap=100)
        
        # Generate embeddings
        all_embeddings, all_contents = concurrent_embed_documents(
            embed_model, split_docs_list, batch_size=50, max_workers=4
        )
        
        # Prepare vectors for upsert
        vectors_to_upsert = [
            (str(idx), embedding, {"text": content})
            for idx, (embedding, content) in enumerate(zip(all_embeddings, all_contents))
        ]
        
        # Upsert to Pinecone
        batch_upsert(pinecone_index, vectors_to_upsert, batch_size=100)
        
        return {"status": "success", "message": f"Ingested {len(vectors_to_upsert)} document chunks"}
        
    except Exception as e:
        print(f"Error ingesting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Endpoint to get an AI response to a question"""
    try:
        # Embed the user's question
        query_embed = embed_model.embed_query(request.question)
        query_embed = [float(val) for val in query_embed]  # Ensure standard floats
        
        # Query Pinecone for relevant documents
        results = pinecone_index.query(
            vector=query_embed,
            top_k=3,
            include_values=False,
            include_metadata=True
        )
        
        # Extract document contents
        doc_contents = []
        for match in results.get('matches', []):
            text = match['metadata'].get('text', '')
            doc_contents.append(text)
        
        context = "\n".join(doc_contents) if doc_contents else "No additional information found."
        
        # Format the system prompt with retrieved content
        formatted_prompt = system_prompt_template.format(context=context)
        
        # Initialize Gemini Flash model
        chat = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )
        
        # Create the conversation prompt
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(formatted_prompt),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        
        # Create the conversation chain
        conversation = LLMChain(
            llm=chat,
            prompt=prompt
        )
        
        # Generate the response
        res = conversation({"question": request.question})
        
        return {
            "question": request.question,
            "answer": res.get('text', ''),
            "context_used": doc_contents,
            "retrieval_count": len(doc_contents)
        }
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Kamwenja TTC Chatbot API is running",
        "model": "gemini-1.5-flash",
        "vector_db": "pinecone",
        "embedding_model": "models/embedding-001"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
