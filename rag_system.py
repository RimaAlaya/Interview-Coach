# rag_system.py - FIXED for Chroma filter format

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from question_bank import QUESTION_BANK
import os


class InterviewRAG:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize RAG system with Chroma vector database"""
        self.persist_directory = persist_directory

        # Use free HuggingFace embeddings (runs locally, no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize or load vector store
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Create or load Chroma vector database"""
        # Check if database already exists
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("📂 Loading existing question database...")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"✅ Loaded {vectorstore._collection.count()} questions from database")
        else:
            print("🔨 Building question database for the first time...")
            # Convert questions to documents
            documents = []
            for i, q in enumerate(QUESTION_BANK):
                # Create rich metadata for better retrieval
                metadata = {
                    "question_id": i,
                    "type": q["type"],
                    "difficulty": q["difficulty"],
                    "category": q["category"],
                    "skills": ", ".join(q["skills"]),
                    "hint": q.get("hint", "")
                }

                # Combine question with metadata for embedding
                content = f"""
                Question: {q['question']}
                Type: {q['type']}
                Difficulty: {q['difficulty']}
                Category: {q['category']}
                Skills: {', '.join(q['skills'])}
                """

                documents.append(Document(
                    page_content=content.strip(),
                    metadata=metadata
                ))

            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"✅ Built database with {len(documents)} questions")

        return vectorstore

    def get_relevant_questions(self, resume_text, job_description,
                               interview_type, difficulty,
                               company_style, num_questions=5):
        """
        Get most relevant questions based on resume, job, and preferences

        FIXED: Proper Chroma filter format with $and operator
        """

        # Build rich query combining all context
        query = f"""
        Resume Context: {resume_text[:500]}

        Job Requirements: {job_description[:500]}

        Interview Type: {interview_type}
        Difficulty Level: {difficulty}
        Company Style: {company_style}

        Find relevant interview questions that match the candidate's background 
        and the job requirements.
        """

        # FIXED: Proper Chroma filter format
        # Use $and for multiple conditions
        filter_dict = {
            "$and": [
                {"type": {"$eq": interview_type}},
                {"difficulty": {"$eq": difficulty}}
            ]
        }

        try:
            # Search for relevant questions
            results = self.vectorstore.similarity_search(
                query=query,
                k=num_questions * 3,  # Get more than needed for filtering
                filter=filter_dict
            )

            # Convert back to question format
            questions = []
            seen_ids = set()

            for doc in results:
                q_id = doc.metadata.get("question_id")
                if q_id is not None and q_id not in seen_ids:
                    seen_ids.add(q_id)
                    question_data = QUESTION_BANK[q_id].copy()
                    questions.append(question_data)

                    if len(questions) >= num_questions:
                        break

            # If we don't have enough questions with filters, search without filters
            if len(questions) < num_questions:
                print(f"⚠️ Only found {len(questions)} questions with filters, searching without filters...")
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=num_questions
                )

                for doc in results:
                    q_id = doc.metadata.get("question_id")
                    if q_id is not None and q_id not in seen_ids:
                        seen_ids.add(q_id)
                        question_data = QUESTION_BANK[q_id].copy()
                        questions.append(question_data)

                        if len(questions) >= num_questions:
                            break

            return questions

        except Exception as e:
            print(f"❌ RAG search error: {str(e)}")
            # Fallback: return questions without vector search
            return self._get_fallback_questions(interview_type, difficulty, num_questions)

    def _get_fallback_questions(self, interview_type, difficulty, num_questions):
        """Fallback method if vector search fails"""
        print("⚠️ Using fallback question selection...")

        # Filter questions by type and difficulty
        filtered = [q for q in QUESTION_BANK
                    if q['type'] == interview_type and q['difficulty'] == difficulty]

        # If not enough, just filter by type
        if len(filtered) < num_questions:
            filtered = [q for q in QUESTION_BANK if q['type'] == interview_type]

        # If still not enough, return any questions
        if len(filtered) < num_questions:
            filtered = QUESTION_BANK

        return filtered[:num_questions]

    def get_follow_up_question(self, previous_question, answer_text,
                               interview_type, difficulty):
        """
        Get a relevant follow-up question based on previous Q&A
        """

        query = f"""
        Previous Question: {previous_question}
        Candidate's Answer: {answer_text[:300]}

        Find a good follow-up question that:
        - Digs deeper into the same topic
        - Or explores a related skill mentioned in the answer
        - Matches the interview type: {interview_type}
        - At difficulty level: {difficulty}
        """

        # FIXED: Proper filter format
        filter_dict = {
            "$and": [
                {"type": {"$eq": interview_type}},
                {"difficulty": {"$eq": difficulty}}
            ]
        }

        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=3,
                filter=filter_dict
            )

            if results:
                q_id = results[0].metadata.get("question_id")
                if q_id is not None:
                    return QUESTION_BANK[q_id].copy()

        except Exception as e:
            print(f"⚠️ Follow-up search error: {str(e)}")

        # Fallback
        return None

    def search_by_skills(self, skills_list, interview_type, difficulty, k=5):
        """
        Search questions by specific skills
        """

        skills_query = ", ".join(skills_list)
        query = f"Questions about: {skills_query}"

        # FIXED: Proper filter format
        filter_dict = {
            "$and": [
                {"type": {"$eq": interview_type}},
                {"difficulty": {"$eq": difficulty}}
            ]
        }

        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )

            questions = []
            for doc in results:
                q_id = doc.metadata.get("question_id")
                if q_id is not None:
                    questions.append(QUESTION_BANK[q_id].copy())

            return questions

        except Exception as e:
            print(f"⚠️ Skills search error: {str(e)}")
            return self._get_fallback_questions(interview_type, difficulty, k)

    def get_company_style_questions(self, company_style, difficulty, k=5):
        """Get questions matching specific company interview style"""

        company_categories = {
            "google": ["google_style", "system_design", "scalability"],
            "amazon": ["amazon_lp", "leadership", "customer_focus"],
            "meta": ["meta_style", "innovation", "impact"],
            "microsoft": ["collaboration", "growth_mindset"],
            "startup": ["startup_style", "versatility", "resourcefulness"],
            "consulting": ["case_study", "analytics", "business_strategy"]
        }

        categories = company_categories.get(company_style, [])

        if categories:
            query = f"Company style: {company_style}. Focus areas: {', '.join(categories)}"

            try:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k
                )

                questions = []
                for doc in results:
                    q_id = doc.metadata.get("question_id")
                    if q_id is not None:
                        questions.append(QUESTION_BANK[q_id].copy())

                return questions

            except Exception as e:
                print(f"⚠️ Company style search error: {str(e)}")
                return []

        return []


# Global instance (initialized once)
_rag_system = None


def get_rag_system():
    """Get or create RAG system singleton"""
    global _rag_system
    if _rag_system is None:
        print("🚀 Initializing RAG system...")
        _rag_system = InterviewRAG()
    return _rag_system


# Test function
if __name__ == "__main__":
    print("🧪 Testing RAG System...")

    rag = get_rag_system()

    # Test query
    resume = "Data scientist with 5 years experience in Python, ML, and NLP"
    job = "Looking for senior ML engineer with deep learning expertise"

    print("\n🔍 Testing question retrieval...")
    questions = rag.get_relevant_questions(
        resume_text=resume,
        job_description=job,
        interview_type="technical",
        difficulty="senior",
        company_style="google",
        num_questions=5
    )

    print(f"\n✅ Retrieved {len(questions)} questions:")
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['difficulty'].upper()}] {q['question'][:80]}...")
        print(f"   Skills: {', '.join(q['skills'][:3])}")

    print("\n✅ RAG system test complete!")