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

# ============================================
# SETUP - Replace with your Groq API key
# ============================================
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # ⚠️ CHANGE THIS!

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize sentiment analysis (for emotion detection)
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
except Exception as e:
    print(f"Warning: Could not load emotion models: {e}")
    sentiment_analyzer = None
    emotion_detector = None


# Wrapper to match IBM Watsonx interface
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

# ============================================
# INTERVIEW CONFIGURATIONS
# ============================================

INTERVIEW_TYPES = {
    "behavioral": {
        "description": "Focus on past experiences and behavioral patterns using STAR method",
        "system_prompt": "You are an expert behavioral interviewer. Focus on situational questions that reveal how candidates handled past challenges. Ask about specific situations, tasks, actions, and results."
    },
    "technical": {
        "description": "Focus on technical skills, problem-solving, and domain knowledge",
        "system_prompt": "You are a technical interviewer. Focus on technical competencies, problem-solving approaches, system design, and hands-on experience with technologies."
    },
    "leadership": {
        "description": "Focus on management skills, team leadership, and decision-making",
        "system_prompt": "You are a senior leadership interviewer. Focus on management experience, team leadership, conflict resolution, strategic thinking, and decision-making abilities."
    },
    "case_study": {
        "description": "Focus on analytical thinking and business problem-solving",
        "system_prompt": "You are a case interview expert. Present business scenarios and assess analytical thinking, problem-solving frameworks, and structured reasoning."
    }
}

DIFFICULTY_LEVELS = {
    "junior": {
        "description": "Entry-level questions (0-2 years experience)",
        "modifier": "Ask basic, foundational questions suitable for candidates with 0-2 years of experience. Focus on fundamental concepts and eagerness to learn."
    },
    "mid": {
        "description": "Intermediate questions (3-5 years experience)",
        "modifier": "Ask intermediate questions suitable for candidates with 3-5 years of experience. Expect practical experience and ability to work independently."
    },
    "senior": {
        "description": "Advanced questions (5+ years experience)",
        "modifier": "Ask advanced questions suitable for candidates with 5+ years of experience. Expect strategic thinking, leadership, and expertise in complex problem-solving."
    }
}

COMPANY_STYLES = {
    "google": {
        "description": "Google-style: Algorithmic thinking, scalability, innovation",
        "focus": "Focus on algorithmic thinking, system scalability, innovative solutions, and data-driven decision making. Emphasize googleyness and leadership qualities."
    },
    "amazon": {
        "description": "Amazon-style: Leadership principles, customer obsession, ownership",
        "focus": "Base questions on Amazon's 16 leadership principles. Focus on customer obsession, ownership, bias for action, and deliver results. Use STAR method extensively."
    },
    "microsoft": {
        "description": "Microsoft-style: Collaboration, growth mindset, technical depth",
        "focus": "Focus on growth mindset, collaboration, technical depth, and inclusive culture. Ask about learning from failures and working across teams."
    },
    "meta": {
        "description": "Meta-style: Impact, moving fast, boldness",
        "focus": "Focus on impact-driven work, moving fast, bold ideas, and being open. Ask about tackling ambiguous problems and shipping products quickly."
    },
    "consulting": {
        "description": "Consulting-style: Case studies, frameworks, structured thinking",
        "focus": "Present business cases and assess structured problem-solving, framework usage, quantitative analysis, and clear communication of recommendations."
    },
    "startup": {
        "description": "Startup-style: Versatility, ownership, scrappiness",
        "focus": "Focus on wearing multiple hats, ownership mentality, resourcefulness with constraints, adaptability, and building from zero to one."
    },
    "general": {
        "description": "General corporate interview style",
        "focus": "Standard professional interview approach covering skills, experience, and cultural fit."
    }
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
    """Get duration of audio file in seconds"""
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
    """Analyze speech quality: filler words, pace, pauses"""
    if not transcript:
        return {}

    # Count filler words
    fillers = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally', 'kind of', 'sort of']
    transcript_lower = transcript.lower()
    filler_count = sum(transcript_lower.count(filler) for filler in fillers)

    # Calculate speaking rate
    duration = get_audio_duration(audio_file_path) if audio_file_path else 0
    word_count = len(transcript.split())
    words_per_minute = (word_count / (duration / 60)) if duration > 0 else 0

    # Determine pace
    if words_per_minute < 100:
        pace = "too slow"
    elif words_per_minute > 180:
        pace = "too fast"
    else:
        pace = "good"

    # Analyze sentence structure
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
    """Analyze emotion and sentiment of answer"""
    if not answer_text or not sentiment_analyzer or not emotion_detector:
        return {"confidence": "N/A", "emotion": "N/A", "feedback": ""}

    try:
        sentiment = sentiment_analyzer(answer_text[:512])[0]  # Limit text length
        emotion = emotion_detector(answer_text[:512])[0]

        confidence_level = "confident" if sentiment['score'] > 0.7 else "somewhat uncertain"

        feedback = ""
        if sentiment['label'] == 'NEGATIVE':
            feedback = "Try to frame your experiences more positively."
        if emotion['label'] in ['fear', 'sadness']:
            feedback += " Your tone seems hesitant. Speak with more confidence."

        return {
            "confidence": confidence_level,
            "emotion": emotion['label'],
            "sentiment_score": round(sentiment['score'], 2),
            "feedback": feedback
        }
    except Exception as e:
        return {"confidence": "N/A", "emotion": "N/A", "feedback": ""}


def save_checkpoint():
    """Save current interview state"""
    global chat_histories, interview_step, resume_summary, job_summary, feedback_history

    checkpoint = {
        "chat_histories": chat_histories,
        "interview_step": interview_step,
        "resume_summary": resume_summary,
        "job_summary": job_summary,
        "feedback_history": feedback_history,
        "timestamp": datetime.now().isoformat()
    }

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_file = f"checkpoints/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    return f"✅ Interview saved to {checkpoint_file}"


def load_checkpoint(checkpoint_file):
    """Load interview state from checkpoint"""
    global chat_histories, interview_step, resume_summary, job_summary, feedback_history

    try:
        with open(checkpoint_file.name, 'r') as f:
            checkpoint = json.load(f)

        chat_histories = checkpoint["chat_histories"]
        interview_step = checkpoint["interview_step"]
        resume_summary = checkpoint["resume_summary"]
        job_summary = checkpoint["job_summary"]
        feedback_history = checkpoint.get("feedback_history", [])

        return "✅ Interview loaded successfully!"
    except Exception as e:
        return f"❌ Error loading checkpoint: {e}"


def save_interview_history(chat_histories, evaluation, scores, resume_name, job_title):
    """Save complete interview to history"""
    os.makedirs("interview_history", exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"interview_history/interview_{timestamp}.json"

    history = {
        "timestamp": datetime.now().isoformat(),
        "candidate": resume_name or "Unknown",
        "position": job_title or "Unknown",
        "questions_answers": chat_histories,
        "evaluation": evaluation,
        "scores": scores,
        "total_questions": len(chat_histories)
    }

    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)

    return filename


# ============================================
# AI AGENTS (Enhanced)
# ============================================

def Resume_Analyst(resume):
    prompt = f"""
    Write a detailed REPORT on the candidate in exactly three paragraphs:

    1. Candidate's background (name if available, education, years of experience)
    2. Key technical and soft skills
    3. Summary of past experiences and achievements

    Resume:
    {resume}
    """
    response = llm_base.chat(
        messages=[
            {"role": "system", "content": "You are an HR expert in reviewing resumes."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']


def Job_Description_Expert(job_description):
    prompt = f"""
    Analyze this job description and provide:
    1. Required technical skills
    2. Required soft skills  
    3. Preferred experience level
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


def Interview_Question_Action(chat_histories, resume_summary, job_summary, interview_type, difficulty, company_style):
    interview_context = INTERVIEW_TYPES[interview_type]["system_prompt"]
    difficulty_context = DIFFICULTY_LEVELS[difficulty]["modifier"]
    company_context = COMPANY_STYLES[company_style]["focus"]

    prompt = f"""
    Based on the interview history, resume, and job requirements, decide the next question action.

    Interview Type: {interview_type}
    Difficulty Level: {difficulty}
    Company Style: {company_style}

    Choose ONE action:
    - Ask about a different skill or experience from the resume
    - Ask a follow-up question to dig deeper into current topic
    - Challenge the candidate with a harder question on same topic

    Answer History:
    {chat_histories}

    Resume Summary:
    {resume_summary}

    Job Requirements:
    {job_summary}

    Context: {interview_context}
    Difficulty: {difficulty_context}
    Company Focus: {company_context}
    """

    response = llm_base.chat(
        messages=[
            {"role": "system", "content": "You are an interview strategy expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']


def Interviewer(resume_summary, job_summary, interview_type, difficulty, company_style, action=None, last=False,
                practice_mode=False):
    interview_context = INTERVIEW_TYPES[interview_type]["system_prompt"]
    difficulty_context = DIFFICULTY_LEVELS[difficulty]["modifier"]
    company_context = COMPANY_STYLES[company_style]["focus"]

    if not last:
        if action is not None:
            hint_instruction = "Also provide a brief hint about what makes a good answer." if practice_mode else ""

            prompt = f"""
            Generate ONE specific interview question based on:

            Action: {action}
            Resume: {resume_summary}
            Job: {job_summary}

            Interview Type: {interview_type}
            {interview_context}

            Difficulty: {difficulty}
            {difficulty_context}

            Company Style: {company_style}
            {company_context}

            {hint_instruction}

            DO NOT explain why you're asking. Just ask the question directly.
            """

            response = llm_base.chat(
                messages=[
                    {"role": "system", "content": "You are an expert interviewer."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content']
        else:
            return "Tell me about yourself and why you're interested in this position."
    else:
        prompt = f"""
        The interview is ending. Provide a warm, professional closing statement.
        Thank the candidate and mention what happens next.
        Be CONCISE (2-3 sentences).

        Candidate Background: {resume_summary[:200]}
        """
        response = llm_base.chat(
            messages=[
                {"role": "system", "content": "You are an expert interviewer."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']


def Real_Time_Feedback(answer_text, question_text, practice_mode=False):
    """Provide immediate feedback on answer quality"""
    if not practice_mode or not answer_text:
        return ""

    prompt = f"""
    Provide BRIEF feedback (2-3 sentences max) on this interview answer:

    Question: {question_text}
    Answer: {answer_text}

    Give:
    - One strength
    - One improvement tip
    - Rating: Strong/Good/Needs Improvement
    """

    response = llm_base.chat(
        messages=[
            {"role": "system", "content": "You are an interview coach providing quick feedback."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content']


def Evaluator(chat_histories, job_summary, interview_type, difficulty):
    prompt = f"""
    Evaluate this interview performance comprehensively:

    Interview Type: {interview_type}
    Difficulty Level: {difficulty}

    Provide evaluation with:
    1. Overall Assessment (Strong Match/Good Match/Needs Improvement/Poor Match)
    2. Strengths (3-4 specific points)
    3. Areas for Improvement (3-4 specific points)
    4. Specific Examples from their answers
    5. Recommendation

    Questions & Answers:
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
    """Generate numerical scores for different competencies"""
    prompt = f"""
    Score this interview on a scale of 0-10 for each category.
    Return ONLY a JSON object, no other text.

    Categories:
    - technical_skills: Technical competency demonstrated
    - communication: Clarity and articulation
    - problem_solving: Analytical thinking and approach
    - cultural_fit: Alignment with role requirements
    - experience_relevance: How well experience matches job
    - overall: Overall performance

    Interview Type: {interview_type}

    Answers:
    {chat_histories}

    Job Requirements:
    {job_summary}

    Format (use this exact structure):
    {{
        "technical_skills": number,
        "communication": number,
        "problem_solving": number,
        "cultural_fit": number,
        "experience_relevance": number,
        "overall": number,
        "justification": "Brief explanation"
    }}
    """

    response = llm_base.chat(
        messages=[
            {"role": "system", "content": "You are an interview scoring expert. Return only JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        # Extract JSON from response
        content = response['choices'][0]['message']['content']
        # Find JSON object in the response
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = content[start:end]
            scores = json.loads(json_str)
            return scores
        else:
            # Fallback scores
            return {
                "technical_skills": 7,
                "communication": 7,
                "problem_solving": 7,
                "cultural_fit": 7,
                "experience_relevance": 7,
                "overall": 7,
                "justification": "Unable to parse detailed scores"
            }
    except:
        return {
            "technical_skills": 7,
            "communication": 7,
            "problem_solving": 7,
            "cultural_fit": 7,
            "experience_relevance": 7,
            "overall": 7,
            "justification": "Scoring error occurred"
        }


# ============================================
# CORE INTERVIEW ENGINE (Enhanced)
# ============================================

def next_question(resume_path, job_str, total_number, interview_type, difficulty, company_style,
                  practice_mode, question_previous="", answer_previous=None, video_input=None):
    global chat_histories, interview_step, resume_summary, job_summary, feedback_history

    # Validate inputs
    if resume_path is None:
        error_msg = "⚠️ Please upload a resume PDF before starting the interview!"
        return gr.update(value=None), gr.update(value=None), gr.update(value="Start Interview"), gr.update(
            value=error_msg)

    if not job_str or job_str.strip() == "":
        error_msg = "⚠️ Please paste a job description before starting the interview!"
        return gr.update(value=None), gr.update(value=None), gr.update(value="Start Interview"), gr.update(
            value=error_msg)

    # Generate summaries (first time only)
    if resume_summary is None:
        resume_text = extract_text_from_pdf(resume_path)
        resume_summary = Resume_Analyst(resume_text)

    if job_summary is None:
        job_summary = Job_Description_Expert(job_str)

    # Transcribe user's answer
    answer_text = ""
    speech_analysis = {}
    emotion_analysis = {}

    if answer_previous:
        answer_text = transcribe_audio_faster_whisper(answer_previous)

        # Analyze speech quality
        speech_analysis = analyze_speech_quality(answer_previous, answer_text)

        # Analyze emotion
        emotion_analysis = analyze_answer_emotion(answer_text)

    # Update chat history
    if interview_step > 0 and question_previous:
        chat_histories[f"Q{interview_step}: {question_previous}"] = {
            "answer": answer_text,
            "speech_analysis": speech_analysis,
            "emotion_analysis": emotion_analysis
        }

    # Get real-time feedback (practice mode only)
    real_time_feedback = ""
    if interview_step > 0 and practice_mode and answer_text:
        real_time_feedback = Real_Time_Feedback(answer_text, question_previous, practice_mode)
        feedback_history.append({
            "question": question_previous,
            "feedback": real_time_feedback,
            "speech_analysis": speech_analysis,
            "emotion_analysis": emotion_analysis
        })

    # Generate next question
    if interview_step < total_number:
        if interview_step == 0:
            action = None
        else:
            # Simplify chat history for action planner
            chat_hist_str = "\n".join([f"Q: {k}\nA: {v['answer']}" for k, v in chat_histories.items()])
            action = Interview_Question_Action(chat_hist_str, resume_summary, job_summary,
                                               interview_type, difficulty, company_style)

        Question_next = Interviewer(resume_summary, job_summary, interview_type, difficulty,
                                    company_style, action, last=False, practice_mode=practice_mode)
    else:
        Question_next = Interviewer(resume_summary, job_summary, interview_type, difficulty,
                                    company_style, action=None, last=True)

    # Build feedback display
    feedback_display = ""
    if real_time_feedback:
        feedback_display = f"### 💡 Quick Feedback\n{real_time_feedback}\n\n"

    if speech_analysis:
        feedback_display += f"""
### 🎤 Speech Analysis
- **Words per minute**: {speech_analysis.get('words_per_minute', 'N/A')} ({speech_analysis.get('pace', 'N/A')} pace)
- **Filler words**: {speech_analysis.get('filler_count', 0)}
- **Answer length**: {speech_analysis.get('word_count', 0)} words
- **Duration**: {speech_analysis.get('duration_seconds', 0)} seconds
"""

    if emotion_analysis and emotion_analysis.get('confidence') != 'N/A':
        feedback_display += f"""
### 😊 Emotion & Confidence
- **Confidence Level**: {emotion_analysis.get('confidence', 'N/A')}
- **Emotion Detected**: {emotion_analysis.get('emotion', 'N/A')}
{emotion_analysis.get('feedback', '')}
"""

    # Evaluate if complete
    if interview_step >= total_number:
        # Generate scores
        chat_hist_str = "\n".join([f"Q: {k}\nA: {v['answer']}" for k, v in chat_histories.items()])
        scores = Score_Interview(chat_hist_str, job_summary, interview_type)

        # Generate evaluation
        evaluation = Evaluator(chat_hist_str, job_summary, interview_type, difficulty)

        # Format scores display
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
"""

        # Save interview history
        try:
            filename = save_interview_history(chat_histories, evaluation, scores,
                                              resume_path.name if resume_path else "Unknown",
                                              "Position")
            scores_display += f"\n\n✅ **Interview saved to**: {filename}"
        except Exception as e:
            scores_display += f"\n\n⚠️ Could not save interview: {e}"

        # Reset for next interview
        chat_histories = {}
        interview_step = 0
        resume_summary = None
        job_summary = None
        feedback_history = []

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
# GRADIO UI (Enhanced)
# ============================================

with gr.Blocks() as demo:
    gr.Markdown("# 🎯 Advanced AI Interview Coach")
    gr.Markdown("### Powered by Groq + Advanced NLP Analysis ⚡")

    with gr.Tabs():
        with gr.TabItem("🎤 Interview"):
            gr.Markdown('## Step 1: Upload Resume & Job Description')
            gr.Markdown('⚠️ **Both are required before starting!**')

            with gr.Row():
                resume_input = gr.File(label="📄 Upload Resume (PDF)", type='filepath')
                job_desc_input = gr.Textbox(label="💼 Job Description", lines=10,
                                            placeholder="Paste the job description here...")

            gr.Markdown('## Step 2: Configure Interview Settings')

            with gr.Row():
                interview_type = gr.Dropdown(
                    choices=list(INTERVIEW_TYPES.keys()),
                    value="behavioral",
                    label="🎭 Interview Type",
                    info="Select the interview focus"
                )
                difficulty = gr.Dropdown(
                    choices=list(DIFFICULTY_LEVELS.keys()),
                    value="mid",
                    label="📈 Difficulty Level",
                    info="Match your experience level"
                )

            with gr.Row():
                company_style = gr.Dropdown(
                    choices=list(COMPANY_STYLES.keys()),
                    value="general",
                    label="🏢 Company Style",
                    info="Interview approach by company type"
                )
                practice_mode = gr.Checkbox(
                    label="🎓 Practice Mode (with hints & real-time feedback)",
                    value=True,
                    info="Disable for realistic simulation"
                )

            num_q_input = gr.Slider(
                label="❓ Number of Questions",
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                info="Total questions in this interview"
            )

            gr.Markdown('## Step 3: Start Interview!')

            interviewer_question = gr.Audio(label="🎙️ Interviewer Question", type="filepath")

            with gr.Row():
                user_answer = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Record Your Answer (Audio)"
                )
                video_input = gr.Video(
                    sources=["webcam"],
                    label="📹 Record Yourself (Video - Optional)",
                    include_audio=True
                )

            start_btn = gr.Button("▶️ Start / Submit Answer", variant="primary", size="lg")

            gr.Markdown("## 📊 Performance Feedback")
            evaluation_textbox = gr.Textbox(label="Real-time Feedback & Final Evaluation",
                                            lines=25)

            # Hidden state to pass previous question
            question_state = gr.State("")


            def process_answer(resume, job, total, itype, diff, company, practice,
                               prev_q, answer_audio, video):
                result = next_question(resume, job, total, itype, diff, company, practice,
                                       prev_q, answer_audio, video)
                # Extract the new question text for next iteration
                new_question = "Question asked"  # Simplified
                return result + (new_question,)


            start_btn.click(
                fn=process_answer,
                inputs=[resume_input, job_desc_input, num_q_input, interview_type,
                        difficulty, company_style, practice_mode, question_state,
                        user_answer, video_input],
                outputs=[interviewer_question, user_answer, start_btn, evaluation_textbox, question_state]
            )

        with gr.TabItem("💾 Save/Load"):
            gr.Markdown("## Save Your Interview Progress")
            gr.Markdown("Use this to pause and resume your interview later.")

            save_btn = gr.Button("💾 Save Current Interview", variant="secondary")
            save_status = gr.Textbox(label="Save Status", interactive=False)

            save_btn.click(
                fn=save_checkpoint,
                outputs=save_status
            )

            gr.Markdown("## Load Previous Interview")
            checkpoint_file = gr.File(label="📁 Upload Checkpoint File", type='filepath')
            load_btn = gr.Button("📂 Load Interview", variant="secondary")
            load_status = gr.Textbox(label="Load Status", interactive=False)

            load_btn.click(
                fn=load_checkpoint,
                inputs=checkpoint_file,
                outputs=load_status
            )

        with gr.TabItem("📚 Interview Guide"):
            gr.Markdown("""
            ## 📖 How to Use This Interview Coach

            ### 🎯 Interview Types

            **Behavioral**: Past experiences using STAR method (Situation, Task, Action, Result)
            - Example: "Tell me about a time you faced a conflict with a team member"

            **Technical**: Technical skills and problem-solving
            - Example: "How would you design a scalable database system?"

            **Leadership**: Management and decision-making
            - Example: "Describe your leadership style and how you motivate teams"

            **Case Study**: Business problem-solving
            - Example: "Our client's revenue dropped 20%. What would you investigate?"

            ### 📈 Difficulty Levels

            - **Junior**: Entry-level (0-2 years) - Foundational knowledge
            - **Mid**: Intermediate (3-5 years) - Practical experience
            - **Senior**: Advanced (5+ years) - Strategic thinking & leadership

            ### 🏢 Company Styles

            - **Google**: Innovation, scalability, algorithmic thinking
            - **Amazon**: Leadership principles, ownership, customer focus
            - **Microsoft**: Collaboration, growth mindset, technical depth
            - **Meta**: Impact, moving fast, bold ideas
            - **Consulting**: Case studies, frameworks, structured thinking
            - **Startup**: Versatility, ownership, resourcefulness

            ### 🎓 Practice vs Real Mode

            **Practice Mode** (Recommended for learning):
            - ✅ Get hints with questions
            - ✅ Real-time feedback after each answer
            - ✅ Speech quality analysis
            - ✅ Emotion detection
            - ✅ See improvement suggestions

            **Real Mode** (For final prep):
            - ❌ No hints
            - ❌ No real-time feedback (only final evaluation)
            - ✅ Realistic interview pressure
            - ✅ Timed experience

            ### 💡 Tips for Great Answers

            1. **Structure**: Use STAR method for behavioral questions
            2. **Specificity**: Give concrete examples with metrics
            3. **Clarity**: Speak clearly at 120-150 words per minute
            4. **Brevity**: Keep answers 1-2 minutes (150-300 words)
            5. **Confidence**: Minimize filler words (um, uh, like)
            6. **Honesty**: Be genuine about challenges and learnings

            ### 🎤 Recording Tips

            - Ensure quiet environment
            - Speak clearly into microphone
            - Take a breath before answering
            - Look at camera if using video
            - Smile and show enthusiasm
            """)

        with gr.TabItem("📊 Performance Metrics"):
            gr.Markdown("""
            ## 🎯 Understanding Your Scores

            ### Score Breakdown (0-10 scale)

            **Technical Skills** (0-10)
            - 9-10: Expert level, comprehensive knowledge
            - 7-8: Strong competency, minor gaps
            - 5-6: Adequate, needs development
            - 0-4: Significant improvement needed

            **Communication** (0-10)
            - Clarity of expression
            - Structure of answers
            - Use of examples
            - Listening and responding appropriately

            **Problem Solving** (0-10)
            - Analytical thinking
            - Approach to challenges
            - Creativity in solutions
            - Logical reasoning

            **Cultural Fit** (0-10)
            - Alignment with company values
            - Team collaboration indicators
            - Work style compatibility
            - Growth mindset

            **Experience Relevance** (0-10)
            - How well past experience matches role
            - Transferable skills demonstrated
            - Career progression logic
            - Industry knowledge

            ### 🎤 Speech Quality Metrics

            **Words Per Minute**:
            - Too Slow: < 100 wpm (shows lack of preparation)
            - Optimal: 120-160 wpm (clear and confident)
            - Too Fast: > 180 wpm (hard to follow)

            **Filler Words**: Aim for < 5 per answer
            - Um, uh, like, you know, basically
            - Shows nervousness or lack of preparation
            - Practice reduces filler usage

            **Answer Length**:
            - Too Short: < 100 words (insufficient detail)
            - Optimal: 150-300 words (comprehensive)
            - Too Long: > 400 words (rambling)

            ### 😊 Emotion Analysis

            The system detects:
            - **Joy/Enthusiasm**: Shows passion (positive)
            - **Confidence**: Indicated by tone and word choice
            - **Fear/Anxiety**: May indicate nervousness
            - **Neutral**: Professional demeanor

            ### 📈 Improvement Over Time

            Track your progress:
            1. Save each interview
            2. Compare scores across sessions
            3. Focus on lowest scoring areas
            4. Repeat same difficulty to see improvement
            5. Gradually increase difficulty level
            """)

        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## 🚀 Advanced AI Interview Coach

            ### Features

            ✅ **5 Specialized AI Agents**
            - Resume Analyst
            - Job Description Expert
            - Interview Strategist
            - Interviewer
            - Evaluator

            ✅ **Multiple Interview Types**
            - Behavioral (STAR method)
            - Technical (Problem-solving)
            - Leadership (Management)
            - Case Study (Business analysis)

            ✅ **Adaptive Difficulty**
            - Junior (Entry-level)
            - Mid (Intermediate)
            - Senior (Advanced)

            ✅ **Company-Specific Styles**
            - FAANG (Google, Amazon, Meta, Microsoft)
            - Consulting
            - Startup
            - General Corporate

            ✅ **Real-Time Analysis**
            - Speech quality (pace, fillers, clarity)
            - Emotion detection (confidence, sentiment)
            - Immediate feedback (practice mode)
            - Numerical scoring

            ✅ **Practice & Real Modes**
            - Practice: Learn with hints and feedback
            - Real: Simulate actual interview pressure

            ✅ **Save/Resume Functionality**
            - Pause interviews anytime
            - Resume from checkpoints
            - Export complete history

            ✅ **Video Recording**
            - Optional video capture
            - Better audio quality
            - Self-review capability

            ### Technology Stack

            - **LLM**: Groq (Llama 3.3-70B)
            - **Speech-to-Text**: Faster Whisper
            - **Text-to-Speech**: Google TTS
            - **Emotion Analysis**: Hugging Face Transformers
            - **UI Framework**: Gradio
            - **PDF Processing**: PyPDF2

            ### Credits

            Built with ❤️ using open-source AI technologies

            ### Version
            v2.0 - Advanced Edition with Multi-Modal Analysis

            ### Support

            For issues or questions, check your saved interview files in:
            - `checkpoints/` - Saved progress
            - `interview_history/` - Completed interviews
            """)

if __name__ == "__main__":
    print("🚀 Starting Advanced Interview Coach...")
    print("📁 Creating necessary directories...")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("interview_history", exist_ok=True)
    print("✅ Ready! Opening browser...")
    demo.launch(share=True)