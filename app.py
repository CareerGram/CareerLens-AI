import streamlit as st
import fitz
import chromadb

from pathlib import Path
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama


# CareerLens AI | Built by Team CareerGram
st.set_page_config(
    page_title="CareerLens AI",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 CareerLens AI")
st.write(
    "Ask questions about your resume and job description. "
    "Get answers grounded in document evidence."
)

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
)

COLLECTION_NAME = "career_documents"


# Load the models and database
@st.cache_resource
def load_embedding_model():
    return TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )


@st.cache_resource
def load_llm():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Download the GGUF model into the models folder."
        )

    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_threads=2,
        n_batch=64,
        verbose=False,
    )


@st.cache_resource
def load_chroma_client():
    return chromadb.Client()


embedding_model = load_embedding_model()
chroma_client = load_chroma_client()


# Read the text from each resume page
def extract_resume_pages(pdf_file):
    pages = []

    pdf_bytes = pdf_file.getvalue()

    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "text": text,
                    "page": page_number,
                })

    return pages


# Break the resume and job description into smaller pieces
def create_document_chunks(resume_pages, job_description):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    documents = []
    metadatas = []
    ids = []

    # Keep the page number so we can refer back to the resume
    for page_data in resume_pages:
        chunks = splitter.split_text(page_data["text"])

        for chunk in chunks:
            index = len(documents)

            documents.append(chunk)
            metadatas.append({
                "source": "resume",
                "page": str(page_data["page"]),
            })
            ids.append(f"resume_{index}")

    # Split the job description and store it separately
    if job_description.strip():
        jd_chunks = splitter.split_text(
            job_description.strip()
        )

        for chunk in jd_chunks:
            index = len(documents)

            documents.append(chunk)
            metadatas.append({
                "source": "job_description",
                "page": "N/A",
            })
            ids.append(f"job_description_{index}")

    return documents, metadatas, ids


# Save the document chunks in ChromaDB for searching
def build_knowledge_base(resume_file, job_description):
    resume_pages = extract_resume_pages(resume_file)

    if not resume_pages:
        raise ValueError(
            "No readable text was found in the resume PDF. "
            "If it is a scanned PDF, OCR may be required."
        )

    if not job_description.strip():
        raise ValueError(
            "Please enter a job description."
        )

    documents, metadatas, ids = create_document_chunks(
        resume_pages,
        job_description,
    )

    if not documents:
        raise ValueError("No document text was found.")

    # Convert each chunk into an embedding
    embeddings = list(
        embedding_model.embed(documents)
    )

    # Start with a fresh collection for the uploaded documents
    try:
        chroma_client.delete_collection(
            COLLECTION_NAME
        )
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add the chunks in small batches
    batch_size = 32

    for start in range(0, len(documents), batch_size):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    return len(documents), len(resume_pages)


# Find the document chunks most relevant to a question
def retrieve_evidence(question, document_filter, top_k=4):
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    question_embedding = list(
        embedding_model.embed([question])
    )[0]

    query_args = {
        "query_embeddings": [question_embedding],
        "n_results": min(
            top_k,
            collection.count()
        ),
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    # Search only the selected document, if needed
    if document_filter == "Resume only":
        query_args["where"] = {
            "source": "resume"
        }

    elif document_filter == "Job description only":
        query_args["where"] = {
            "source": "job_description"
        }

    results = collection.query(**query_args)

    evidence = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = metadata["source"]

        if source == "resume":
            reference = f"Resume, page {metadata['page']}"
        else:
            reference = "Job description"

        evidence.append({
            "text": document,
            "source": source,
            "reference": reference,
            "distance": distance,
        })

    return evidence


# Use the retrieved text to answer the question
def generate_answer(question, evidence):
    llm = load_llm()

    evidence_text = "\n\n".join(
        [
            f"[Source: {item['reference']}]\n"
            f"{item['text']}"
            for item in evidence
        ]
    )

    prompt = f"""
You are CareerLens AI, an evidence-based career assistant.

Answer the user's question using only the evidence below.

Rules:
1. Use simple English.
2. Do not invent skills, qualifications, experience, or requirements.
3. If the evidence does not contain the answer, clearly say so.
4. Distinguish resume evidence from job-description requirements.
5. When comparing documents, explain what matches and what is missing.
6. Mention the source, such as Resume page 2 or Job description.
7. A skill mentioned in a resume is not always proof of practical experience.

EVIDENCE:
{evidence_text}

USER QUESTION:
{question}

Write a clear, helpful answer with source references.
"""

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful RAG assistant. "
                    "Never make up facts not supported by "
                    "the supplied evidence."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=220,
        temperature=0.1,
    )

    return response["choices"][0]["message"]["content"]


# Upload the documents and prepare them for searching
st.subheader("Step 1: Add your documents")

resume_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"],
)

job_description = st.text_area(
    "Paste the job description",
    height=220,
    placeholder=(
        "Paste the job title, responsibilities, "
        "required skills, qualifications, and experience..."
    ),
)

if st.button(
    "Build / Update Knowledge Base",
    type="primary",
):
    if resume_file is None:
        st.error("Please upload your resume PDF.")

    elif not job_description.strip():
        st.error("Please paste the job description.")

    else:
        with st.spinner(
            "Extracting text, creating embeddings, and indexing documents..."
        ):
            try:
                chunk_count, page_count = build_knowledge_base(
                    resume_file,
                    job_description,
                )

                st.session_state["career_index_ready"] = True
                st.session_state["career_index_chunks"] = chunk_count
                st.session_state["career_resume_pages"] = page_count

                st.success(
                    f"Knowledge base ready! "
                    f"{chunk_count} chunks indexed from "
                    f"{page_count} resume pages and the job description."
                )

            except Exception as error:
                st.session_state["career_index_ready"] = False
                st.error(f"Could not build the knowledge base: {error}")


# Ask questions and display the answers
st.divider()
st.subheader("Step 2: Ask a question")

document_filter = st.selectbox(
    "Which documents should CareerLens AI search?",
    [
        "Both documents",
        "Resume only",
        "Job description only",
    ],
)

question = st.text_input(
    "Your question",
    placeholder=(
        "Example: What are the key skills required for this job?"
    ),
)

if st.button("Ask CareerLens AI"):
    if not st.session_state.get("career_index_ready", False):
        st.warning(
            "Please upload both documents and build the knowledge base first."
        )

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching documents and generating an answer..."):
                evidence = retrieve_evidence(
                    question,
                    document_filter,
                )

                answer = generate_answer(
                    question,
                    evidence,
                )

            st.subheader("Answer")
            st.write(answer)

            # Show the text used to generate the answer
            with st.expander("View retrieved evidence"):
                for number, item in enumerate(evidence, start=1):
                    st.markdown(
                        f"**Evidence {number} — {item['reference']}**"
                    )
                    st.write(item["text"])

        except Exception as error:
            st.error(f"Something went wrong: {error}")


# A few details about the app
with st.sidebar:
    st.header("About CareerLens AI")
    st.write(
        "CareerLens AI uses Retrieval-Augmented Generation "
        "(RAG) to answer questions from your resume and "
        "job description."
    )

    st.markdown("**Available features**")
    st.write("• Resume question answering")
    st.write("• Job-description question answering")
    st.write("• Resume and job comparison")
    st.write("• Retrieved evidence with source references")

    st.caption(
        "The GGUF language model runs locally. "
        "Answers depend on the quality of the uploaded documents "
        "and the retrieved evidence."
    )
