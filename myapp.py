from groq import Groq
from gtts import gTTS
from faster_whisper import WhisperModel
import PyPDF2
import gradio as gr
import json
import os
from datetime import datetime
from transformers import pipeline
import re
from collections import Counter

# Import RAG system
from rag_system import get_rag_system

# ============================================
# SETUP - Replace with your Groq API key
# ============================================
GROQ_API_KEY = "put your api here"  # ⚠️ CHANGE THIS!

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize sentiment analysis
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
except Exception as e:
    print(f"Warning: Could not load emotion models: {e}")
    sentiment_analyzer = None
    emotion_detector = None

# Initialize RAG system (will load on first use)
rag_system = None


class LLMWrapper:
    def __init__(self, client):
        self.client = client

    def chat(self, messages, max_tokens=8000):
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return {
            'choices': [{
                'message': {
                    'content': response.choices[0].message.content
                }
            }]
        }


llm_base = LLMWrapper(groq_client)

# Global variables
chat_histories = {}
interview_step = 0
resume_summary = None
job_summary = None
feedback_history = []
checkpoint_data = {}
selected_questions = []  # Store RAG-selected questions
current_question_index = 0

# ============================================
# INTERVIEW CONFIGURATIONS
# ============================================

INTERVIEW_TYPES = {
    "behavioral": {
        "description": "Focus on past experiences and behavioral patterns using STAR method",
        "system_prompt": "You are an expert behavioral interviewer. Focus on situational questions that reveal how candidates handled past challenges."
    },
    "technical": {
        "description": "Focus on technical skills, problem-solving, and domain knowledge",
        "system_prompt": "You are a technical interviewer. Focus on technical competencies, problem-solving approaches, and hands-on experience."
    },
    "leadership": {
        "description": "Focus on management skills, team leadership, and decision-making",
        "system_prompt": "You are a senior leadership interviewer. Focus on management experience, team leadership, and strategic thinking."
    },
    "case_study": {
        "description": "Focus on analytical thinking and business problem-solving",
        "system_prompt": "You are a case interview expert. Present business scenarios and assess analytical thinking and structured reasoning."
    }
}

DIFFICULTY_LEVELS = {
    "junior": {
        "description": "Entry-level questions (0-2 years experience)",
        "modifier": "Ask basic, foundational questions suitable for candidates with 0-2 years of experience."
    },
    "mid": {
        "description": "Intermediate questions (3-5 years experience)",
        "modifier": "Ask intermediate questions suitable for candidates with 3-5 years of experience."
    },
    "senior": {
        "description": "Advanced questions (5+ years experience)",
        "modifier": "Ask advanced questions suitable for candidates with 5+ years of experience."
    }
}

COMPANY_STYLES = {
    "google": {"description": "Google-style: Algorithmic thinking, scalability, innovation"},
    "amazon": {"description": "Amazon-style: Leadership principles, customer obsession"},
    "microsoft": {"description": "Microsoft-style: Collaboration, growth mindset"},
    "meta": {"description": "Meta-style: Impact, moving fast, boldness"},
    "consulting": {"description": "Consulting-style: Case studies, frameworks"},
    "startup": {"description": "Startup-style: Versatility, ownership, scrappiness"},
    "general": {"description": "General corporate interview style"}
}


# ============================================
# UTILITY FUNCTIONS
# ============================================

def extract_text_from_pdf(pdf_file_path):
    reader = PyPDF2.PdfReader(pdf_file_path.name)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def text_to_speech_file(text_input):
    audio_file_path = "temp_voice.mp3"
    tts = gTTS(text=text_input, lang='en')
    tts.save(audio_file_path)
    return audio_file_path


def transcribe_audio_faster_whisper(audio_file_path: str, model_size: str = "base") -> str:
    if audio_file_path is None:
        return ""
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_file_path, beam_size=5)
        full_transcript = [segment.text for segment in segments]
        return "".join(full_transcript).strip()
    except Exception as e:
        return f"Transcription error: {e}"


def get_audio_duration(audio_file_path):
    try:
        import wave
        with wave.open(audio_file_path, 'rb') as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            duration = frames / float(rate)
            return duration
    except:
        return 0


def analyze_speech_quality(audio_file_path, transcript):
    if not transcript:
        return {}

    fillers = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally']
    transcript_lower = transcript.lower()
    filler_count = sum(transcript_lower.count(filler) for filler in fillers)

    duration = get_audio_duration(audio_file_path) if audio_file_path else 0
    word_count = len(transcript.split())
    words_per_minute = (word_count / (duration / 60)) if duration > 0 else 0

    if words_per_minute < 100:
        pace = "too slow"
    elif words_per_minute > 180:
        pace = "too fast"
    else:
        pace = "good"

    sentences = [s.strip() for s in transcript.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

    return {
        "filler_count": filler_count,
        "words_per_minute": round(words_per_minute, 1),
        "pace": pace,
        "word_count": word_count,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "duration_seconds": round(duration, 1)
    }


def analyze_answer_emotion(answer_text):
    if not answer_text or not sentiment_analyzer or not emotion_detector:
        return {"confidence": "N/A", "emotion": "N/A", "feedback": ""}

    try:
        sentiment = sentiment_analyzer(answer_text[:512])[0]
        emotion = emotion_detector(answer_text[:512])[0]

        confidence_level = "confident" if sentiment['score'] > 0.7 else "somewhat uncertain"

        feedback = ""
        if sentiment['label'] == 'NEGATIVE':
            feedback = "Try to frame your experiences more positively."
        if emotion['label'] in ['fear', 'sadness']:
            feedback += " Speak with more confidence."

        return {
            "confidence": confidence_level,
            "emotion": emotion['label'],
            "sentiment_score": round(sentiment['score'], 2),
            "feedback": feedback
        }
    except Exception as e:
        return {"confidence": "N/A", "emotion": "N/A", "feedback": ""}


def save_checkpoint():
    global chat_histories, interview_step, resume_summary, job_summary, feedback_history, selected_questions

    checkpoint = {
        "chat_histories": chat_histories,
        "interview_step": interview_step,
        "resume_summary": resume_summary,
        "job_summary": job_summary,
        "feedback_history": feedback_history,
        "selected_questions": selected_questions,
        "timestamp": datetime.now().isoformat()
    }

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_file = f"checkpoints/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    return f"✅ Interview saved to {checkpoint_file}"


def load_checkpoint(checkpoint_file):
    global chat_histories, interview_step, resume_summary, job_summary, feedback_history, selected_questions

    try:
        with open(checkpoint_file.name, 'r') as f:
            checkpoint = json.load(f)

        chat_histories = checkpoint["chat_histories"]
        interview_step = checkpoint["interview_step"]
        resume_summary = checkpoint["resume_summary"]
        job_summary = checkpoint["job_summary"]
        feedback_history = checkpoint.get("feedback_history", [])
        selected_questions = checkpoint.get("selected_questions", [])

        return "✅ Interview loaded successfully!"
    except Exception as e:
        return f"❌ Error loading checkpoint: {e}"


# ============================================
# AI AGENTS WITH RAG
# ============================================

def Resume_Analyst(resume):
    prompt = f"""
    Analyze this resume and extract:
    1. Key technical skills (list them clearly)
    2. Years of experience
    3. Main areas of expertise
    4. Notable achievements

    Resume:
    {resume}

    Be specific about skills for matching with job requirements.
    """
    response = llm_base.chat(
        messages=[
            {"role": "system", "content": "You are an HR expert. Extract structured information."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']


def Job_Description_Expert(job_description):
    prompt = f"""
    Analyze this job description and extract:
    1. Required technical skills (list clearly)
    2. Required soft skills
    3. Experience level needed
    4. Key responsibilities

    Job Description:
    {job_description}
    """
    response = llm_base.chat(
        messages=[
            {"role": "system", "content": "You are a job analysis expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']


def Real_Time_Feedback(answer_text, question_text, practice_mode=False):
    if not practice_mode or not answer_text:
        return ""

    prompt = f"""
    Provide BRIEF feedback (2-3 sentences) on this answer:

    Question: {question_text}
    Answer: {answer_text}

    Give:
    - One strength
    - One improvement tip
    - Rating: Strong/Good/Needs Improvement
    """

    response = llm_base.chat(
        messages=[
            {"role": "system", "content": "You are an interview coach."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content']


def Evaluator(chat_histories, job_summary, interview_type, difficulty):
    prompt = f"""
    Evaluate this interview comprehensively:

    Interview Type: {interview_type}
    Difficulty: {difficulty}

    Provide:
    1. Overall Assessment
    2. Strengths (3-4 points)
    3. Areas for Improvement (3-4 points)
    4. Specific Examples
    5. Recommendation

    Q&A:
    {chat_histories}

    Job Requirements:
    {job_summary}
    """

    response = llm_base.chat(
        messages=[
            {"role": "system", "content": "You are an expert interview evaluator."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000
    )
    return response['choices'][0]['message']['content']


def Score_Interview(chat_histories, job_summary, interview_type):
    prompt = f"""
    Score this interview 0-10 for each category. Return ONLY JSON.

    Categories:
    - technical_skills
    - communication
    - problem_solving
    - cultural_fit
    - experience_relevance
    - overall

    Format:
    {{
        "technical_skills": number,
        "communication": number,
        "problem_solving": number,
        "cultural_fit": number,
        "experience_relevance": number,
        "overall": number,
        "justification": "Brief explanation"
    }}

    Answers:
    {chat_histories}

    Job:
    {job_summary}
    """

    response = llm_base.chat(
        messages=[
            {"role": "system", "content": "You are an interview scoring expert. Return only JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        content = response['choices'][0]['message']['content']
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            scores = json.loads(json_str)
            return scores
    except:
        pass

    return {
        "technical_skills": 7, "communication": 7, "problem_solving": 7,
        "cultural_fit": 7, "experience_relevance": 7, "overall": 7,
        "justification": "Scoring error occurred"
    }


# ============================================
# CORE INTERVIEW ENGINE WITH RAG
# ============================================

def next_question(resume_path, job_str, total_number, interview_type, difficulty,
                  company_style, practice_mode, question_previous="", answer_previous=None,
                  video_input=None):
    global chat_histories, interview_step, resume_summary, job_summary
    global feedback_history, rag_system, selected_questions, current_question_index

    # Validate inputs
    if resume_path is None:
        return gr.update(value=None), gr.update(value=None), gr.update(value="Start Interview"), \
            gr.update(value="⚠️ Please upload a resume PDF!")

    if not job_str or job_str.strip() == "":
        return gr.update(value=None), gr.update(value=None), gr.update(value="Start Interview"), \
            gr.update(value="⚠️ Please paste a job description!")

    # Initialize RAG system (first time only)
    if rag_system is None:
        rag_system = get_rag_system()

    # Generate summaries (first time only)
    if resume_summary is None:
        resume_text = extract_text_from_pdf(resume_path)
        resume_summary = Resume_Analyst(resume_text)

    if job_summary is None:
        job_summary = Job_Description_Expert(job_str)

    # Get RAG-selected questions (first time only)
    if not selected_questions:
        print("🎯 Using RAG to select relevant questions...")
        resume_text = extract_text_from_pdf(resume_path)
        selected_questions = rag_system.get_relevant_questions(
            resume_text=resume_text,
            job_description=job_str,
            interview_type=interview_type,
            difficulty=difficulty,
            company_style=company_style,
            num_questions=total_number
        )
        print(f"✅ Selected {len(selected_questions)} relevant questions")

    # Transcribe answer
    answer_text = ""
    speech_analysis = {}
    emotion_analysis = {}

    if answer_previous:
        answer_text = transcribe_audio_faster_whisper(answer_previous)
        speech_analysis = analyze_speech_quality(answer_previous, answer_text)
        emotion_analysis = analyze_answer_emotion(answer_text)

    # Update chat history
    if interview_step > 0 and question_previous:
        chat_histories[f"Q{interview_step}: {question_previous}"] = {
            "answer": answer_text,
            "speech_analysis": speech_analysis,
            "emotion_analysis": emotion_analysis
        }

    # Real-time feedback
    real_time_feedback = ""
    if interview_step > 0 and practice_mode and answer_text:
        real_time_feedback = Real_Time_Feedback(answer_text, question_previous, practice_mode)
        feedback_history.append({
            "question": question_previous,
            "feedback": real_time_feedback,
            "speech_analysis": speech_analysis,
            "emotion_analysis": emotion_analysis
        })

    # Get next question from RAG-selected list
    if interview_step < total_number:
        if interview_step < len(selected_questions):
            question_data = selected_questions[interview_step]
            Question_next = question_data["question"]

            # Add hint if practice mode
            if practice_mode and question_data.get("hint"):
                Question_next += f"\n\n💡 Hint: {question_data['hint']}"
        else:
            Question_next = "Tell me more about your experience."
    else:
        Question_next = "Thank you for completing the interview. You'll receive detailed feedback shortly."

    # Build feedback display
    feedback_display = ""
    if real_time_feedback:
        feedback_display = f"### 💡 Quick Feedback\n{real_time_feedback}\n\n"

    if speech_analysis:
        feedback_display += f"""
### 🎤 Speech Analysis
- **Pace**: {speech_analysis.get('words_per_minute', 'N/A')} wpm ({speech_analysis.get('pace', 'N/A')})
- **Filler words**: {speech_analysis.get('filler_count', 0)}
- **Length**: {speech_analysis.get('word_count', 0)} words ({speech_analysis.get('duration_seconds', 0)}s)
"""

    if emotion_analysis and emotion_analysis.get('confidence') != 'N/A':
        feedback_display += f"""
### 😊 Confidence & Emotion
- **Confidence**: {emotion_analysis.get('confidence', 'N/A')}
- **Emotion**: {emotion_analysis.get('emotion', 'N/A')}
{emotion_analysis.get('feedback', '')}
"""

    # Evaluate if complete
    if interview_step >= total_number:
        chat_hist_str = "\n".join([f"Q: {k}\nA: {v['answer']}" for k, v in chat_histories.items()])
        scores = Score_Interview(chat_hist_str, job_summary, interview_type)
        evaluation = Evaluator(chat_hist_str, job_summary, interview_type, difficulty)

        scores_display = f"""
## 📊 Your Scores

- **Technical Skills**: {scores.get('technical_skills', 'N/A')}/10
- **Communication**: {scores.get('communication', 'N/A')}/10
- **Problem Solving**: {scores.get('problem_solving', 'N/A')}/10
- **Cultural Fit**: {scores.get('cultural_fit', 'N/A')}/10
- **Experience Relevance**: {scores.get('experience_relevance', 'N/A')}/10
- **Overall Score**: {scores.get('overall', 'N/A')}/10

**Justification**: {scores.get('justification', '')}

---

## 📝 Detailed Evaluation

{evaluation}

---

## 🎯 Questions Asked (Selected by AI)

"""
        for i, q in enumerate(selected_questions[:total_number], 1):
            scores_display += f"{i}. **[{q['difficulty'].upper()}]** {q['question'][:100]}...\n"

        # Reset
        chat_histories = {}
        interview_step = 0
        resume_summary = None
        job_summary = None
        feedback_history = []
        selected_questions = []
        current_question_index = 0

        final_evaluation = scores_display
    else:
        final_evaluation = feedback_display if feedback_display else "Interview in progress..."

    # Convert to audio
    question_audio_path = text_to_speech_file(Question_next)
    interview_step += 1

    return (
        gr.update(value=question_audio_path),
        gr.update(value=None),
        gr.update(value="Submit Answer"),
        gr.update(value=final_evaluation)
    )


# ============================================
# GRADIO UI
# ============================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 AI Interview Coach with RAG")
    gr.Markdown("### Powered by Groq + LangChain + Chroma Vector DB ⚡")

    with gr.Tabs():
        with gr.TabItem("🎤 Interview"):
            gr.Markdown('## Step 1: Upload Materials')

            with gr.Row():
                resume_input = gr.File(label="📄 Resume (PDF)", type='filepath')
                job_desc_input = gr.Textbox(label="💼 Job Description", lines=10,
                                            placeholder="Paste job description...")

            gr.Markdown('## Step 2: Configure Interview')

            with gr.Row():
                interview_type = gr.Dropdown(
                    choices=list(INTERVIEW_TYPES.keys()),
                    value="behavioral",
                    label="🎭 Interview Type"
                )
                difficulty = gr.Dropdown(
                    choices=list(DIFFICULTY_LEVELS.keys()),
                    value="mid",
                    label="📈 Difficulty"
                )

            with gr.Row():
                company_style = gr.Dropdown(
                    choices=list(COMPANY_STYLES.keys()),
                    value="general",
                    label="🏢 Company Style"
                )
                practice_mode = gr.Checkbox(
                    label="🎓 Practice Mode (hints + feedback)",
                    value=True
                )

            num_q_input = gr.Slider(1, 10, value=5, step=1, label="❓ Questions")

            gr.Markdown('## Step 3: Interview')

            interviewer_question = gr.Audio(label="🎙️ Question", type="filepath")

            with gr.Row():
                user_answer = gr.Audio(sources=["microphone"], type="filepath",
                                       label="🎤 Your Answer")
                video_input = gr.Video(sources=["webcam"], label="📹 Video (Optional)",
                                       include_audio=True)

            start_btn = gr.Button("▶️ Start / Submit", variant="primary", size="lg")

            gr.Markdown("## 📊 Feedback & Results")
            evaluation_textbox = gr.Textbox(label="Real-time Analysis", lines=25)

            question_state = gr.State("")


            def process_answer(resume, job, total, itype, diff, company, practice,
                               prev_q, answer_audio, video):
                result = next_question(resume, job, total, itype, diff, company, practice,
                                       prev_q, answer_audio, video)
                new_question = "Question asked"
                return result + (new_question,)


            start_btn.click(
                fn=process_answer,
                inputs=[resume_input, job_desc_input, num_q_input, interview_type,
                        difficulty, company_style, practice_mode, question_state,
                        user_answer, video_input],
                outputs=[interviewer_question, user_answer, start_btn,
                         evaluation_textbox, question_state]
            )

        with gr.TabItem("💾 Save/Load"):
            gr.Markdown("## Save Progress")
            save_btn = gr.Button("💾 Save Interview")
            save_status = gr.Textbox(label="Status")
            save_btn.click(fn=save_checkpoint, outputs=save_status)

            gr.Markdown("## Load Interview")
            checkpoint_file = gr.File(label="📁 Checkpoint File", type='filepath')
            load_btn = gr.Button("📂 Load")
            load_status = gr.Textbox(label="Status")
            load_btn.click(fn=load_checkpoint, inputs=checkpoint_file, outputs=load_status)

        with gr.TabItem("📚 Guide"):
            gr.Markdown("""
            ## 🎯 How It Works

            ### RAG-Powered Question Selection

            This system uses **Retrieval Augmented Generation (RAG)** to select the most relevant 
            interview questions for you:

            1. **Vector Database**: 80+ questions stored with embeddings
            2. **Smart Matching**: Questions selected based on:
               - Your resume skills
               - Job requirements
               - Interview type & difficulty
               - Company interview style
            3. **Chroma DB**: Free, local vector database (no API key needed)
            4. **HuggingFace Embeddings**: Free embeddings running locally

            ### Question Bank Categories

            - **Behavioral**: STAR method, past experiences
            - **Technical**: ML, data science, system design
            - **Leadership**: Management, decision-making
            - **Case Study**: Business problems, analytics
            - **Company-Specific**: Amazon LP, Google-style, etc.

            ### Benefits of RAG

            ✅ Personalized questions matching YOUR background
            ✅ Relevant to the SPECIFIC job you're applying for
            ✅ No generic questions
            ✅ Adaptive difficulty
            ✅ 100% free (no API costs)

            ### Tips

            1. Upload detailed resume for better matching
            2. Paste complete job description
            3. Practice mode shows hints from question bank
            4. Questions are selected before interview starts
            5. Each interview gets fresh question selection
            """)

        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## 🚀 Technology Stack

            ### AI & ML
            - **LLM**: Groq (Llama 3.3-70B) - Fast inference
            - **RAG**: LangChain + Chroma - Question retrieval
            - **Embeddings**: HuggingFace (all-MiniLM-L6-v2) - Free
            - **Speech**: Faster Whisper + gTTS
            - **Emotion**: Transformers (DistilBERT)

            ### Vector Database
            - **Chroma**: Local vector store (no cloud needed)
            - **80+ Questions**: Pre-embedded and indexed
            - **Similarity Search**: Finds most relevant questions
            - **Metadata Filtering**: By type, difficulty, skills

            ### Features

            ✅ RAG-powered question selection
            ✅ Multi-agent AI system
            ✅ Real-time speech analysis
            ✅ Emotion detection
            ✅ Practice & real modes
            ✅ Save/resume functionality
            ✅ 100% free to use

            ### Version
            v3.0 - RAG Edition

            ### Cost
            **$0** - Everything runs locally except Groq API (free tier)
            """)

if __name__ == "__main__":
    print("🚀 Starting AI Interview Coach with RAG...")
    print("📁 Creating directories...")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("interview_history", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)
    print("✅ Ready!")
    demo.launch(share=True)