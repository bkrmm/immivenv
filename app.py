import streamlit as st

# --- LangChain components ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# =============================
# Streamlit setup
# =============================
st.set_page_config(page_title="📘 Gemini RAG Chatbot", layout="wide")
st.title("📘 RAG Chatbot — Gemini for Answers, MiniLM for Embeddings")

GOOGLE_API_KEY = ""

# =============================
# Load or create FAISS vector store
# =============================

@st.cache_resource
def load_vector_store():
    loader = DirectoryLoader("data", glob="*.md", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    modl_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=modl_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =============================
# Gemini Model
# =============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# =============================
# Prompt + Runnable Chain (Modern LangChain API)
# =============================
systemprompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the provided context to answer the question clearly.
If the answer isn't found, say so honestly.

Context:
# Comprehensive Ranking System Lookup Table

## Core/Human Capital Factors;
You will earn points as if you  **don’t**  have a spouse or partner if:
-   they are not coming with you to Canada, or.
-   they are a Canadian citizen or permanent resident.
 
|*_Factors_*|_*Points with Spouse*_  |_*Points without Spouse*_|
|--|--|--|
|Age|100|110|
|Level of Education|140|150|
|Official Language Proficiency|150|160|
|Canadian Work Experience|70|80|

## Detailed Points Breakdown
-   With a spouse or common-law partner: Maximum 460 points total
-   Without a spouse or common-law partner: Maximum 500 points total

|*_Age_*| _*Points with Spouse*_  |_*Points without Spouse*_|
|--|--|--|
|17years or less|0|0
|18years|90|99
|19years|95|105
|20-29years|100|110
|30years|95|105
|31years|90|99
|32years|85|94
|33years|80|88
|34years|75|83
|35years|70|77
|36years|65|72
|37years|60|66
|38years|55|61
|39years|50|55
|40years|45|50
|41years|35|39
|42years|25|28
|43years|15|17
|44years|5|6
|45years or more|0|0|

## Level of Education Points Breakdown
|*_Level of Education_*| _*Points with Spouse*_  |_*Points without Spouse*_|
|--|--|--|
|Less than secondary school|0|0|
|Secondary diploma (high school graduation)|28|30|
|One-year degree, diploma or certificate from a university, college, trade or technical school, or other institute|84|90|
|Two-year program at a university, college, trade or technical school, or other institute|91|98|
|Bachelor's degree OR a three or more year program at a university, college, trade or technical school, or other institute|112|120|
|Two or more certificates, diplomas, or degrees. One must be for a program of three or more years|119|128|
|Master's degree, OR professional degree needed to practice in a licensed profession (For “professional degree,” the degree program must have been in: medicine, veterinary medicine, dentistry, optometry, law, chiropractic medicine, or pharmacy.)|126|135|
|Doctoral level university degree (Ph.D.)|140|150|

## Official Language Proficiency Points Breakdown
Maximum points for each ability (reading, writing, speaking and listening):
-   32 with a spouse or common-law partner
-   34 without a spouse or common-law partner

_Canadian Language Benchmark *CLB* Level_|_With a Spouse_|_Without a Spouse_|
|--|--|--|
|Less than CLB 4|0|0
|CLB 4 or 5|6|6|
|CLB 6|8|9|
|CLB 7|16|17|
|CLB 8|22|23|
|CLB 9|29|31|
|CLB 10 or more|32|34|

### Second Language Points Breakdown
_Canadian Language Benchmark *CLB* Level_|_With a Spouse_|_Without a Spouse_|
--|--|--|
|CLB 4 or less|0|0|
|CLB 5 or 6|1|1|
|CLB 7 or 8|3|3|
|CLB 9 or more|6|6|

## Canadian Work Experience
Canadian Work Experience| With a Spouse| Without a Spouse|
|--|--|--
|None or less than a year|0|0|
|1 year|35|40|
|2 years|46|53|
|3 years|56|64|
|4 years|63|72|
|5 years or more|70|80|

## Skill Transferability Factors;
|Education | Points
--|--
With good official languages proficiency and a post-secondary degree|50
With Canadian work experience and a post-secondary degree|50
___
Foreign Work Experience|Points
--|--
With good official language proficiency and foreign work experience| 50
With Canadian Work Experience and Foreign Work Experience | 50
---
Certificate of Qualification | Points |
--|--
With good/strong official languages proficiency and a certificate of qualification | 50

## Additional Points
|Factor| Points
--|--
|Brother or sister living in Canada (18 years or older, citizen or permanent resident)|15
|French language skills| 50
|Post-secondary education in Canada|30
|Provincial or territorial nomination|600
                                                
                                                
Question:
{question}
""")

# Build RAG chain manually (modern pattern)
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | systemprompt
    | llm
)

# =============================
# Streamlit Chat Interface
# =============================
if "history" not in st.session_state:
    st.session_state["history"] = []

user_query = st.chat_input("Ask something about your markdown files...")

if user_query:
    response = rag_chain.invoke(user_query)
    answer = response.content
    st.session_state["history"].append((user_query, answer))

for q, a in st.session_state["history"]:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

