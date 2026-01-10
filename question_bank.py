# question_bank.py - Question database for RAG system

QUESTION_BANK = [
    # BEHAVIORAL QUESTIONS
    # Leadership & Management
    {
        "question": "Tell me about a time when you had to lead a team through a difficult project. What was your approach?",
        "type": "behavioral",
        "difficulty": "senior",
        "category": "leadership",
        "skills": ["leadership", "team management", "project management"],
        "hint": "Use STAR method: describe the situation, your leadership approach, specific actions, and measurable results."
    },
    {
        "question": "Describe a situation where you had to motivate an underperforming team member. What did you do?",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "leadership",
        "skills": ["coaching", "motivation", "communication"],
        "hint": "Show empathy and your coaching approach. Include the outcome."
    },
    {
        "question": "Tell me about a time you had to make an unpopular decision. How did you handle it?",
        "type": "behavioral",
        "difficulty": "senior",
        "category": "leadership",
        "skills": ["decision making", "communication", "stakeholder management"],
        "hint": "Demonstrate courage and clear communication. Show how you built buy-in."
    },

    # Conflict Resolution
    {
        "question": "Describe a time when you disagreed with a team member. How did you resolve it?",
        "type": "behavioral",
        "difficulty": "junior",
        "category": "teamwork",
        "skills": ["conflict resolution", "communication", "collaboration"],
        "hint": "Show you can handle disagreements professionally and find win-win solutions."
    },
    {
        "question": "Tell me about a time when you had to work with a difficult stakeholder. What was your strategy?",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "stakeholder_management",
        "skills": ["stakeholder management", "communication", "influence"],
        "hint": "Show emotional intelligence and ability to build relationships despite challenges."
    },

    # Problem Solving
    {
        "question": "Describe a situation where you identified a significant problem that others missed. What did you do?",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "problem_solving",
        "skills": ["analytical thinking", "initiative", "attention to detail"],
        "hint": "Demonstrate proactive thinking and impact of your discovery."
    },
    {
        "question": "Tell me about the most complex problem you've solved. Walk me through your approach.",
        "type": "behavioral",
        "difficulty": "senior",
        "category": "problem_solving",
        "skills": ["analytical thinking", "strategic thinking", "technical skills"],
        "hint": "Show systematic problem-solving approach and technical depth."
    },
    {
        "question": "Describe a time when you had to solve a problem with limited resources. How did you approach it?",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "problem_solving",
        "skills": ["resourcefulness", "creativity", "prioritization"],
        "hint": "Show ability to be scrappy and deliver results despite constraints."
    },

    # Failure & Learning
    {
        "question": "Tell me about a project that failed. What did you learn?",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "growth_mindset",
        "skills": ["resilience", "learning agility", "accountability"],
        "hint": "Show ownership, reflection, and how you applied lessons learned."
    },
    {
        "question": "Describe a time when you received critical feedback. How did you respond?",
        "type": "behavioral",
        "difficulty": "junior",
        "category": "growth_mindset",
        "skills": ["receptiveness", "growth mindset", "self-improvement"],
        "hint": "Demonstrate openness to feedback and commitment to growth."
    },

    # Innovation & Improvement
    {
        "question": "Tell me about a time when you improved a process or system. What was your approach?",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "innovation",
        "skills": ["process improvement", "analytical thinking", "initiative"],
        "hint": "Quantify the improvement and show systematic approach to optimization."
    },
    {
        "question": "Describe a situation where you implemented a creative solution to a problem.",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "innovation",
        "skills": ["creativity", "problem solving", "innovation"],
        "hint": "Show thinking outside the box and positive impact of innovation."
    },

    # TECHNICAL QUESTIONS - DATA SCIENCE & ML
    # Machine Learning Fundamentals
    {
        "question": "Explain the difference between supervised and unsupervised learning. Give examples of when you'd use each.",
        "type": "technical",
        "difficulty": "junior",
        "category": "ml_fundamentals",
        "skills": ["machine learning", "supervised learning", "unsupervised learning"],
        "hint": "Define both clearly, give real-world use cases, mention specific algorithms."
    },
    {
        "question": "What is overfitting and how do you prevent it? Describe techniques you've used.",
        "type": "technical",
        "difficulty": "mid",
        "category": "ml_fundamentals",
        "skills": ["machine learning", "model training", "regularization"],
        "hint": "Explain the concept, mention regularization, cross-validation, and early stopping."
    },
    {
        "question": "Walk me through how you would approach a classification problem from scratch.",
        "type": "technical",
        "difficulty": "mid",
        "category": "ml_workflow",
        "skills": ["machine learning", "data science", "problem solving"],
        "hint": "Cover data exploration, feature engineering, model selection, evaluation, and deployment."
    },
    {
        "question": "Explain bias-variance tradeoff. How do you balance it in practice?",
        "type": "technical",
        "difficulty": "mid",
        "category": "ml_fundamentals",
        "skills": ["machine learning", "model optimization", "statistical thinking"],
        "hint": "Define both terms, explain tradeoff, give practical examples from your experience."
    },

    # Deep Learning & NLP
    {
        "question": "How would you build a sentiment analysis system? Walk through your architecture choices.",
        "type": "technical",
        "difficulty": "mid",
        "category": "nlp",
        "skills": ["nlp", "deep learning", "system design"],
        "hint": "Discuss preprocessing, model options (BERT, distilBERT), training approach, and evaluation."
    },
    {
        "question": "Explain how transformers work. Why are they better than RNNs for NLP?",
        "type": "technical",
        "difficulty": "senior",
        "category": "nlp",
        "skills": ["nlp", "deep learning", "transformers"],
        "hint": "Explain attention mechanism, parallelization benefits, and long-range dependencies."
    },
    {
        "question": "How would you implement a RAG (Retrieval Augmented Generation) system? What are the key components?",
        "type": "technical",
        "difficulty": "senior",
        "category": "llm",
        "skills": ["llm", "rag", "vector databases", "system design"],
        "hint": "Cover embeddings, vector store, retrieval, prompt engineering, and LLM integration."
    },
    {
        "question": "Explain the difference between fine-tuning and prompt engineering for LLMs. When would you use each?",
        "type": "technical",
        "difficulty": "mid",
        "category": "llm",
        "skills": ["llm", "prompt engineering", "fine-tuning"],
        "hint": "Compare costs, effort, use cases, and give examples from your experience."
    },

    # MLOps & Deployment
    {
        "question": "How do you monitor machine learning models in production? What metrics do you track?",
        "type": "technical",
        "difficulty": "senior",
        "category": "mlops",
        "skills": ["mlops", "monitoring", "production systems"],
        "hint": "Discuss model drift, data drift, performance metrics, logging, and alerting systems."
    },
    {
        "question": "Describe your approach to versioning and deploying ML models. What tools do you use?",
        "type": "technical",
        "difficulty": "mid",
        "category": "mlops",
        "skills": ["mlops", "deployment", "version control"],
        "hint": "Mention MLflow, DVC, Docker, CI/CD pipelines, and A/B testing strategies."
    },
    {
        "question": "How would you handle model retraining in production? What triggers a retrain?",
        "type": "technical",
        "difficulty": "senior",
        "category": "mlops",
        "skills": ["mlops", "automation", "system design"],
        "hint": "Discuss performance degradation, data drift detection, automated pipelines, and validation."
    },
    {
        "question": "Explain how you would containerize and deploy a machine learning model using Docker.",
        "type": "technical",
        "difficulty": "mid",
        "category": "mlops",
        "skills": ["mlops", "docker", "deployment"],
        "hint": "Walk through Dockerfile creation, dependencies, API setup (FastAPI), and deployment."
    },

    # Data Engineering & Processing
    {
        "question": "How do you handle imbalanced datasets? Give specific techniques you've used.",
        "type": "technical",
        "difficulty": "mid",
        "category": "data_processing",
        "skills": ["data science", "machine learning", "data preprocessing"],
        "hint": "Mention SMOTE, undersampling, class weights, and evaluation metrics for imbalanced data."
    },
    {
        "question": "Explain your feature engineering process. How do you decide which features to create?",
        "type": "technical",
        "difficulty": "mid",
        "category": "feature_engineering",
        "skills": ["feature engineering", "data science", "domain knowledge"],
        "hint": "Discuss domain knowledge, correlation analysis, feature importance, and iterative approach."
    },
    {
        "question": "How would you design a data pipeline for training ML models at scale?",
        "type": "technical",
        "difficulty": "senior",
        "category": "data_engineering",
        "skills": ["data engineering", "pipeline design", "scalability"],
        "hint": "Cover data ingestion, transformation, validation, storage, and orchestration tools."
    },

    # System Design & Architecture
    {
        "question": "Design a recommendation system for an e-commerce platform. What approach would you take?",
        "type": "technical",
        "difficulty": "senior",
        "category": "system_design",
        "skills": ["system design", "machine learning", "scalability"],
        "hint": "Discuss collaborative filtering, content-based, hybrid approaches, cold start, and scaling."
    },
    {
        "question": "How would you build a real-time fraud detection system? Walk through the architecture.",
        "type": "technical",
        "difficulty": "senior",
        "category": "system_design",
        "skills": ["system design", "real-time systems", "machine learning"],
        "hint": "Cover feature engineering, model choice, low latency requirements, and feedback loops."
    },

    # TECHNICAL - SOFTWARE ENGINEERING
    {
        "question": "Explain the difference between SQL and NoSQL databases. When would you use each?",
        "type": "technical",
        "difficulty": "junior",
        "category": "databases",
        "skills": ["sql", "databases", "system design"],
        "hint": "Compare ACID vs BASE, schema flexibility, scalability, and give use case examples."
    },
    {
        "question": "How do you optimize a slow SQL query? Walk through your debugging process.",
        "type": "technical",
        "difficulty": "mid",
        "category": "databases",
        "skills": ["sql", "performance optimization", "debugging"],
        "hint": "Mention EXPLAIN, indexing, query restructuring, and avoiding common pitfalls."
    },
    {
        "question": "Explain RESTful API design principles. How do you structure endpoints?",
        "type": "technical",
        "difficulty": "mid",
        "category": "api_design",
        "skills": ["api design", "rest", "software engineering"],
        "hint": "Cover HTTP methods, resource naming, status codes, versioning, and best practices."
    },
    {
        "question": "How do you ensure code quality in a team environment? What practices do you follow?",
        "type": "technical",
        "difficulty": "mid",
        "category": "software_engineering",
        "skills": ["code quality", "testing", "collaboration"],
        "hint": "Mention code reviews, testing, linting, CI/CD, documentation, and pair programming."
    },

    # CASE STUDY QUESTIONS
    {
        "question": "A company's user engagement dropped 30% last month. How would you investigate and address this?",
        "type": "case_study",
        "difficulty": "senior",
        "category": "analytics",
        "skills": ["analytics", "problem solving", "business acumen"],
        "hint": "Use structured approach: clarify metrics, segment users, check for changes, hypothesis testing."
    },
    {
        "question": "You need to build a churn prediction model. Walk me through your entire approach from problem definition to deployment.",
        "type": "case_study",
        "difficulty": "senior",
        "category": "ml_project",
        "skills": ["machine learning", "project management", "business impact"],
        "hint": "Cover business objective, data collection, modeling, evaluation, deployment, and monitoring."
    },
    {
        "question": "How would you measure the success of a new feature in a mobile app?",
        "type": "case_study",
        "difficulty": "mid",
        "category": "analytics",
        "skills": ["analytics", "metrics", "experimentation"],
        "hint": "Define success metrics, design A/B test, statistical significance, and long-term monitoring."
    },
    {
        "question": "A client wants to reduce customer support costs by 50% using AI. What would you propose?",
        "type": "case_study",
        "difficulty": "senior",
        "category": "business_strategy",
        "skills": ["ai strategy", "business acumen", "solution design"],
        "hint": "Consider chatbots, ticket routing, sentiment analysis, knowledge base, and ROI calculation."
    },

    # LEADERSHIP QUESTIONS
    {
        "question": "How do you prioritize when you have multiple high-priority projects?",
        "type": "leadership",
        "difficulty": "mid",
        "category": "prioritization",
        "skills": ["prioritization", "time management", "stakeholder management"],
        "hint": "Show framework for prioritization (impact/effort, urgency/importance) and communication."
    },
    {
        "question": "Describe your approach to mentoring junior team members.",
        "type": "leadership",
        "difficulty": "senior",
        "category": "mentorship",
        "skills": ["mentorship", "coaching", "leadership"],
        "hint": "Share specific mentoring philosophy, examples, and how you measure growth."
    },
    {
        "question": "How do you handle technical debt while delivering features?",
        "type": "leadership",
        "difficulty": "senior",
        "category": "technical_leadership",
        "skills": ["technical leadership", "decision making", "balance"],
        "hint": "Show ability to balance short-term delivery with long-term code quality."
    },
    {
        "question": "Tell me about a time you had to deliver bad news to stakeholders. How did you approach it?",
        "type": "leadership",
        "difficulty": "mid",
        "category": "communication",
        "skills": ["communication", "stakeholder management", "transparency"],
        "hint": "Show honesty, preparation of alternatives, and managing expectations."
    },

    # COMPANY-SPECIFIC STYLE QUESTIONS
    # Amazon Leadership Principles
    {
        "question": "Tell me about a time when you had to make a decision with incomplete information. (Amazon: Bias for Action)",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "amazon_lp",
        "skills": ["decision making", "bias for action", "risk taking"],
        "hint": "Show calculated risk-taking and bias toward action over analysis paralysis."
    },
    {
        "question": "Describe a time when you went above and beyond for a customer. (Amazon: Customer Obsession)",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "amazon_lp",
        "skills": ["customer focus", "ownership", "initiative"],
        "hint": "Demonstrate putting customer needs first and taking ownership."
    },
    {
        "question": "Tell me about a time you simplified a complex process. (Amazon: Invent and Simplify)",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "amazon_lp",
        "skills": ["simplification", "innovation", "efficiency"],
        "hint": "Show ability to find simpler solutions and challenge the status quo."
    },

    # Google Style
    {
        "question": "Design a system that can handle 1 billion requests per day. How would you scale it?",
        "type": "technical",
        "difficulty": "senior",
        "category": "google_style",
        "skills": ["system design", "scalability", "distributed systems"],
        "hint": "Discuss load balancing, caching, database sharding, and monitoring at scale."
    },
    {
        "question": "How would you approach a problem that has never been solved before?",
        "type": "behavioral",
        "difficulty": "senior",
        "category": "google_style",
        "skills": ["innovation", "research", "problem solving"],
        "hint": "Show structured approach to novel problems and ability to break down complexity."
    },

    # Meta Style
    {
        "question": "Tell me about a time you moved fast and broke things. What did you learn?",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "meta_style",
        "skills": ["speed", "learning", "risk taking"],
        "hint": "Show comfort with fast-paced environment and learning from mistakes."
    },
    {
        "question": "Describe a time when you challenged the conventional way of doing something.",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "meta_style",
        "skills": ["innovation", "boldness", "influence"],
        "hint": "Demonstrate willingness to question status quo and drive change."
    },

    # Startup Style
    {
        "question": "Tell me about a time you wore multiple hats and delivered across different domains.",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "startup_style",
        "skills": ["versatility", "ownership", "adaptability"],
        "hint": "Show comfort with ambiguity and ability to learn quickly across domains."
    },
    {
        "question": "Describe a time you built something from zero to one with limited resources.",
        "type": "behavioral",
        "difficulty": "mid",
        "category": "startup_style",
        "skills": ["resourcefulness", "building", "scrappiness"],
        "hint": "Demonstrate scrappy execution and ability to deliver despite constraints."
    },
]


def get_questions_by_filters(interview_type=None, difficulty=None, category=None, skills=None):
    """Filter questions based on criteria"""
    filtered = QUESTION_BANK

    if interview_type:
        filtered = [q for q in filtered if q["type"] == interview_type]

    if difficulty:
        filtered = [q for q in filtered if q["difficulty"] == difficulty]

    if category:
        filtered = [q for q in filtered if q["category"] == category]

    if skills:
        filtered = [q for q in filtered if any(skill in q["skills"] for skill in skills)]

    return filtered