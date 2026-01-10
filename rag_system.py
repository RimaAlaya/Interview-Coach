# rag_system.py - RAG system using Chroma for question retrieval

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

        Args:
            resume_text: Candidate's resume text
            job_description: Job posting text
            interview_type: Type of interview (behavioral, technical, etc.)
            difficulty: Difficulty level (junior, mid, senior)
            company_style: Company interview style
            num_questions: Number of questions to retrieve

        Returns:
            List of question dictionaries
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

        # Build filter for metadata
        filter_dict = {
            "type": interview_type,
            "difficulty": difficulty
        }

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

        return questions

    def get_follow_up_question(self, previous_question, answer_text,
                               interview_type, difficulty):
        """
        Get a relevant follow-up question based on previous Q&A

        Args:
            previous_question: The question that was just asked
            answer_text: The candidate's answer
            interview_type: Type of interview
            difficulty: Difficulty level

        Returns:
            Follow-up question dictionary
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

        filter_dict = {
            "type": interview_type,
            "difficulty": difficulty
        }

        results = self.vectorstore.similarity_search(
            query=query,
            k=3,
            filter=filter_dict
        )

        if results:
            q_id = results[0].metadata.get("question_id")
            if q_id is not None:
                return QUESTION_BANK[q_id].copy()

        # Fallback
        return None

    def search_by_skills(self, skills_list, interview_type, difficulty, k=5):
        """
        Search questions by specific skills

        Args:
            skills_list: List of skills to search for
            interview_type: Type of interview
            difficulty: Difficulty level
            k: Number of results

        Returns:
            List of relevant questions
        """

        skills_query = ", ".join(skills_list)
        query = f"Questions about: {skills_query}"

        filter_dict = {
            "type": interview_type,
            "difficulty": difficulty
        }

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