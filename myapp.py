from groq import Groq
from gtts import gTTS
from faster_whisper import WhisperModel
import PyPDF2
import gradio as gr
import json
import os
import tempfile
from datetime import datetime
from transformers import pipeline
import re

# Import modules
from rag_system import get_rag_system
from job_scraper import JobScraper
from analytics_dashboard import InterviewAnalytics
from video_handler import VideoInterviewManager

# ============================================
# SETUP
# ============================================
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # ⚠️ CHANGE THIS!

groq_client = Groq(api_key=GROQ_API_KEY)
job_scraper = JobScraper()
analytics = InterviewAnalytics()
video_manager = VideoInterviewManager()  # NEW: Video manager

# Initialize sentiment analysis
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
except Exception as e:
    print(f"Warning: Could not load emotion models: {e}")
    sentiment_analyzer = None
    emotion_detector = None

rag_system = None


# ============================================
# AUDIO HANDLER
# ============================================

class AudioHandler:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.audio_counter = 0

    def text_to_speech_file(self, text_input, lang='en'):
        try:
            if not text_input or text_input.strip() == "":
                return None

            if len(text_input) > 5000:
                text_input = text_input[:5000] + "..."

            self.audio_counter += 1
            audio_file_path = os.path.join(self.temp_dir, f"interview_q_{self.audio_counter}.mp3")

            if os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path)
                except:
                    pass

            print(f"🎤 Generating TTS: {text_input[:60]}...")
            tts = gTTS(text=text_input, lang=lang, slow=False)
            tts.save(audio_file_path)

            if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) < 100:
                return None

            print(f"✅ TTS OK: {os.path.getsize(audio_file_path)} bytes")
            return audio_file_path

        except Exception as e:
            print(f"❌ TTS Error: {str(e)}")
            return None


audio_handler = AudioHandler()


def text_to_speech_file(text_input):
    return audio_handler.text_to_speech_file(text_input)


# ============================================
# LLM WRAPPER
# ============================================

class LLMWrapper:
    def __init__(self, client):
        self.client = client

    def chat(self, messages, max_tokens=8000):
        try:
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
        except Exception as e:
            print(f"❌ LLM Error: {str(e)}")
            return {'choices': [{'message': {'content': f"Error: {str(e)}"}}]}


llm_base = LLMWrapper(groq_client)

# ============================================
# GLOBAL STATE
# ============================================

chat_histories = {}
interview_step = 0
resume_summary = None
job_summary = None
feedback_history = []
selected_questions = []
interview_start_time = None
scraped_job_data = None
video_recording_enabled = False  # NEW

# ============================================
# CONFIGURATIONS
# ============================================

INTERVIEW_TYPES = {
    "behavioral": {"description": "Past experiences using STAR method"},
    "technical": {"description": "Technical skills and problem-solving"},
    "leadership": {"description": "Management and decision-making"},
    "case_study": {"description": "Business analysis and frameworks"}
}

DIFFICULTY_LEVELS = {
    "junior": {"description": "Entry-level (0-2 years)"},
    "mid": {"description": "Intermediate (3-5 years)"},
    "senior": {"description": "Advanced (5+ years)"}
}

COMPANY_STYLES = {
    "google": {"description": "Google-style"},
    "amazon": {"description": "Amazon-style"},
    "microsoft": {"description": "Microsoft-style"},
    "meta": {"description": "Meta-style"},
    "consulting": {"description": "Consulting-style"},
    "startup": {"description": "Startup-style"},
    "general": {"description": "General corporate"}
}


# ============================================
# JOB SCRAPER
# ============================================

def scrape_job_from_url(url):
    if not url or url.strip() == "":
        return None, "⚠️ Please enter a job URL"

    try:
        print(f"🔍 Scraping: {url}")
        result = job_scraper.scrape_job(url)

        if 'error' in result and result.get('title') == 'Could not scrape':
            return None, f"❌ {result['error']}\n\n💡 Try pasting manually."

        formatted = job_scraper.format_job_for_interview(result)

        success_msg = f"""
✅ **Successfully scraped!**

**📋 Position:** {result['title']}
**🏢 Company:** {result['company']}
**📍 Location:** {result['location']}
**🛠️ Skills:** {', '.join(result['skills'][:5])}
"""

        global scraped_job_data
        scraped_job_data = result

        return formatted, success_msg

    except Exception as e:
        return None, f"❌ Scraping failed: {str(e)}"


# ============================================
# UTILITY FUNCTIONS
# ============================================

def extract_text_from_pdf(pdf_file_path):
    try:
        reader = PyPDF2.PdfReader(pdf_file_path.name)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ PDF Error: {str(e)}")
        return ""


def transcribe_audio_faster_whisper(audio_file_path, model_size="base"):
    if audio_file_path is None:
        return ""
    try:
        if not os.path.exists(audio_file_path):
            return "Error: Audio file not found"

        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:
            return "Error: Audio too short"

        print(f"🎤 Transcribing...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_file_path, beam_size=5)
        full_transcript = [segment.text for segment in segments]
        result = "".join(full_transcript).strip()

        return result if result else "Error: Could not transcribe"

    except Exception as e:
        print(f"❌ Transcription Error: {str(e)}")
        return f"Error: {str(e)}"


def get_audio_duration(audio_file_path):
    try:
        import wave
        with wave.open(audio_file_path, 'rb') as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            return frames / float(rate)
    except:
        return 0


def analyze_speech_quality(audio_file_path, transcript):
    if not transcript:
        return {}

    fillers = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
    filler_count = sum(transcript.lower().count(f) for f in fillers)

    duration = get_audio_duration(audio_file_path) if audio_file_path else 0
    word_count = len(transcript.split())
    wpm = (word_count / (duration / 60)) if duration > 0 else 0

    pace = "good" if 120 <= wpm <= 160 else ("too slow" if wpm < 120 else "too fast")

    return {
        "filler_count": filler_count,
        "words_per_minute": round(wpm, 1),
        "pace": pace,
        "word_count": word_count,
        "duration_seconds": round(duration, 1)
    }


def analyze_answer_emotion(answer_text):
    if not answer_text or not sentiment_analyzer:
        return {"confidence": "N/A", "emotion": "N/A"}

    try:
        sentiment = sentiment_analyzer(answer_text[:512])[0]
        emotion = emotion_detector(answer_text[:512])[0] if emotion_detector else None

        return {
            "confidence": "confident" if sentiment['score'] > 0.7 else "uncertain",
            "emotion": emotion['label'] if emotion else "N/A"
        }
    except:
        return {"confidence": "N/A", "emotion": "N/A"}


# ============================================
# AI AGENTS (simplified for space)
# ============================================

def Resume_Analyst(resume):
    prompt = f"Extract key skills and experience:\n{resume[:2000]}"
    response = llm_base.chat([
        {"role": "system", "content": "You are an HR expert."},
        {"role": "user", "content": prompt}
    ])
    return response['choices'][0]['message']['content']


def Job_Description_Expert(job_description):
    prompt = f"Extract required skills:\n{job_description[:2000]}"
    response = llm_base.chat([
        {"role": "system", "content": "You are a job expert."},
        {"role": "user", "content": prompt}
    ])
    return response['choices'][0]['message']['content']


def Real_Time_Feedback(answer_text, question_text, practice_mode=False):
    if not practice_mode or not answer_text:
        return ""

    prompt = f"Brief feedback:\nQ: {question_text}\nA: {answer_text}"
    response = llm_base.chat([
        {"role": "system", "content": "Interview coach."},
        {"role": "user", "content": prompt}
    ], max_tokens=200)
    return response['choices'][0]['message']['content']


def Score_Interview(chat_histories, job_summary, interview_type):
    prompt = f"""Score 0-10 JSON:
{{"technical_skills": 7, "communication": 7, "problem_solving": 7, "cultural_fit": 7, "overall": 7}}
Answers: {chat_histories[:1000]}"""

    response = llm_base.chat([
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": prompt}
    ])

    try:
        content = response['choices'][0]['message']['content']
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(content[start:end])
    except:
        pass

    return {"technical_skills": 7, "communication": 7, "problem_solving": 7,
            "cultural_fit": 7, "overall": 7}


# ============================================
# CORE INTERVIEW ENGINE WITH VIDEO
# ============================================

def next_question(resume_path, job_str, total_number, interview_type, difficulty,
                  company_style, practice_mode, enable_video, question_previous,
                  answer_audio, answer_video):
    global chat_histories, interview_step, resume_summary, job_summary
    global feedback_history, rag_system, selected_questions, interview_start_time
    global video_recording_enabled

    # Validation
    if resume_path is None:
        return (gr.update(value=None), gr.update(value=None), gr.update(value=None),
                gr.update(value="Start Interview"),
                gr.update(value="⚠️ **Upload resume!**"), question_previous)

    if not job_str or job_str.strip() == "":
        return (gr.update(value=None), gr.update(value=None), gr.update(value=None),
                gr.update(value="Start Interview"),
                gr.update(value="⚠️ **Add job description!**"), question_previous)

    # Start session
    if interview_step == 0:
        interview_start_time = datetime.now()
        video_recording_enabled = enable_video

        # Start video session if enabled
        if enable_video:
            job_title = scraped_job_data.get('title', 'Position') if scraped_job_data else 'Position'
            resume_name = os.path.basename(resume_path.name) if resume_path else 'Resume'
            video_manager.start_new_session(resume_name, job_title, interview_type)
            print("📹 Video recording enabled")

    # Initialize RAG
    if rag_system is None:
        rag_system = get_rag_system()

    # Summaries
    if resume_summary is None:
        resume_text = extract_text_from_pdf(resume_path)
        resume_summary = Resume_Analyst(resume_text)

    if job_summary is None:
        job_summary = Job_Description_Expert(job_str)

    # Get questions
    if not selected_questions:
        resume_text = extract_text_from_pdf(resume_path)
        selected_questions = rag_system.get_relevant_questions(
            resume_text=resume_text,
            job_description=job_str,
            interview_type=interview_type,
            difficulty=difficulty,
            company_style=company_style,
            num_questions=total_number
        )
        print(f"✅ Selected {len(selected_questions)} questions")

    # Process answer
    answer_text = ""
    speech_analysis = {}

    # Prefer video audio over separate audio
    audio_to_transcribe = answer_video if (answer_video and enable_video) else answer_audio

    if audio_to_transcribe:
        answer_text = transcribe_audio_faster_whisper(audio_to_transcribe)
        speech_analysis = analyze_speech_quality(audio_to_transcribe, answer_text)

        # Save video if enabled
        if enable_video and answer_video and interview_step > 0:
            video_manager.save_video_segment(
                video_file_path=answer_video,
                question_number=interview_step,
                question_text=question_previous,
                answer_transcript=answer_text,
                duration=speech_analysis.get('duration_seconds', 0)
            )

    # Update history
    if interview_step > 0 and question_previous:
        chat_histories[f"Q{interview_step}: {question_previous}"] = {
            "answer": answer_text,
            "speech_analysis": speech_analysis
        }

    # Feedback
    real_time_feedback = ""
    if interview_step > 0 and practice_mode and answer_text:
        real_time_feedback = Real_Time_Feedback(answer_text, question_previous, practice_mode)

    # Next question or finish
    if interview_step < total_number:
        if interview_step < len(selected_questions):
            q_data = selected_questions[interview_step]
            Question_next = q_data["question"]
            if practice_mode and q_data.get("hint"):
                Question_next += f"\n\n💡 {q_data['hint']}"
        else:
            Question_next = "Tell me about your experience."
    else:
        Question_next = "Interview complete!"

    # Build feedback
    feedback_display = ""
    if real_time_feedback:
        feedback_display = f"### 💡 Feedback\n{real_time_feedback}\n\n"

    if speech_analysis:
        feedback_display += f"""
### 🎤 Speech
- **Pace**: {speech_analysis.get('words_per_minute', 0)} wpm
- **Fillers**: {speech_analysis.get('filler_count', 0)}
- **Length**: {speech_analysis.get('word_count', 0)} words
"""

    # Finalize
    if interview_step >= total_number:
        chat_hist_str = "\n".join([f"Q: {k}\nA: {v['answer']}" for k, v in chat_histories.items()])
        scores = Score_Interview(chat_hist_str, job_summary, interview_type)

        duration = (datetime.now() - interview_start_time).total_seconds() if interview_start_time else 0

        # Finalize video session
        if video_recording_enabled:
            session_id = video_manager.finalize_session(scores, "Interview completed")
            feedback_display += f"\n\n📹 **Video saved! Session: {session_id}**\n"
            feedback_display += "Check 'Video Playback' tab to review.\n"

        # Log analytics
        analytics.log_interview({
            'date': datetime.now().isoformat(),
            'type': interview_type,
            'difficulty': difficulty,
            'scores': scores,
            'speech_metrics': speech_analysis,
            'duration': duration,
            'questions_count': total_number
        })

        feedback_display += f"""
## 📊 Final Scores
- **Technical**: {scores.get('technical_skills', 0)}/10
- **Communication**: {scores.get('communication', 0)}/10
- **Overall**: {scores.get('overall', 0)}/10

✅ Logged to analytics!
"""

        # Reset
        chat_histories = {}
        interview_step = 0
        resume_summary = None
        job_summary = None
        selected_questions = []
        interview_start_time = None
        video_recording_enabled = False

    else:
        if not feedback_display:
            feedback_display = "Answer the question above..."

    # Generate audio
    question_audio_path = text_to_speech_file(Question_next)

    interview_step += 1

    return (
        gr.update(value=question_audio_path),
        gr.update(value=None),  # Clear audio
        gr.update(value=None),  # Clear video
        gr.update(value="Submit Answer"),
        gr.update(value=feedback_display),
        Question_next
    )


# ============================================
# VIDEO PLAYBACK FUNCTIONS
# ============================================

def list_video_sessions():
    """Get list of recorded sessions"""
    sessions = video_manager.get_all_sessions()

    if not sessions:
        return "No recorded sessions yet.", None

    # Format as markdown table
    table = "| Date | Job | Type | Videos | Score |\n"
    table += "|------|-----|------|--------|-------|\n"

    for s in sessions:
        date = s['start_time'][:10] if s['start_time'] else 'N/A'
        table += f"| {date} | {s['job_title'][:20]} | {s['interview_type']} | {s['num_videos']} | {s['final_score']}/10 |\n"

    # Create dropdown choices
    choices = [f"{s['session_id']} - {s['job_title']}" for s in sessions]

    return table, gr.update(choices=choices, value=choices[0] if choices else None)


def load_session_for_playback(session_choice):
    """Load a session for playback"""
    if not session_choice:
        return "Select a session", None, None

    session_id = session_choice.split(' - ')[0]
    session_data = video_manager.load_session(session_id)

    if not session_data:
        return "Session not found", None, None

    # Build session info
    info = f"""
## 📹 Session: {session_id}

**Date:** {session_data['start_time'][:19]}
**Job:** {session_data['job_title']}
**Type:** {session_data['interview_type']}
**Duration:** {session_data['total_duration']:.1f}s
**Videos:** {len(session_data['video_segments'])}

### Questions:
"""

    for i, segment in enumerate(session_data['video_segments'], 1):
        info += f"\n{i}. {segment['question_text'][:60]}..."

    # Get first video
    first_video = None
    if session_data['video_segments']:
        first_video = session_data['video_segments'][0]['video_path']

    # Create question choices
    q_choices = [f"Q{seg['question_number']}: {seg['question_text'][:50]}"
                 for seg in session_data['video_segments']]

    return info, first_video, gr.update(choices=q_choices, value=q_choices[0] if q_choices else None)


def load_specific_video(session_choice, question_choice):
    """Load specific question video"""
    if not session_choice or not question_choice:
        return None, "Select session and question"

    session_id = session_choice.split(' - ')[0]
    q_num = int(question_choice.split(':')[0].replace('Q', ''))

    session_data = video_manager.load_session(session_id)
    if not session_data:
        return None, "Session not found"

    # Find video
    for segment in session_data['video_segments']:
        if segment['question_number'] == q_num:
            video_path = segment['video_path']

            # Build info
            info = f"""
### Question {q_num}

**Q:** {segment['question_text']}

**Answer:** {segment['answer_transcript'][:300]}...

**Duration:** {segment.get('duration', 0):.1f}s
"""
            return video_path, info

    return None, "Video not found"


# ============================================
# ANALYTICS
# ============================================

def show_analytics_dashboard():
    fig = analytics.create_progress_dashboard()
    insights = analytics.get_insights()
    return fig, insights


# ============================================
# GRADIO UI
# ============================================

demo = gr.Blocks()

with demo:
    gr.Markdown("""
    # 🎯 AI Interview Coach Pro
    ### 🎬 Now with Video Recording & Playback!
    """)

    with gr.Tabs():
        with gr.TabItem("🎤 Interview"):
            gr.Markdown('## Step 1: Materials')

            resume_input = gr.File(label="📄 Resume (PDF)", type='filepath')

            with gr.Tabs():
                with gr.TabItem("✍️ Paste"):
                    job_desc_input = gr.Textbox(label="💼 Job Description", lines=8)

                with gr.TabItem("🌐 Scrape"):
                    job_url_input = gr.Textbox(label="🔗 Job URL")
                    scrape_btn = gr.Button("🔍 Scrape", variant="primary")
                    scrape_status = gr.Markdown("")

                    scrape_btn.click(
                        fn=scrape_job_from_url,
                        inputs=[job_url_input],
                        outputs=[job_desc_input, scrape_status]
                    )

            gr.Markdown('## Step 2: Settings')

            with gr.Row():
                interview_type = gr.Dropdown(
                    choices=list(INTERVIEW_TYPES.keys()),
                    value="behavioral",
                    label="🎭 Type"
                )
                difficulty = gr.Dropdown(
                    choices=list(DIFFICULTY_LEVELS.keys()),
                    value="mid",
                    label="📈 Level"
                )

            with gr.Row():
                company_style = gr.Dropdown(
                    choices=list(COMPANY_STYLES.keys()),
                    value="general",
                    label="🏢 Style"
                )
                practice_mode = gr.Checkbox(label="🎓 Practice", value=True)

            with gr.Row():
                num_q_input = gr.Slider(1, 10, value=5, step=1, label="❓ Questions")
                enable_video_recording = gr.Checkbox(label="📹 Record Video", value=True)

            gr.Markdown('## Step 3: Interview')

            interviewer_question = gr.Audio(label="🎙️ Question", type="filepath")

            with gr.Row():
                user_answer_audio = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Audio")
                user_answer_video = gr.Video(sources=["webcam"], label="📹 Video (if enabled)", include_audio=True)

            start_btn = gr.Button("▶️ Start / Submit", variant="primary", size="lg")
            evaluation_textbox = gr.Textbox(label="📊 Feedback", lines=15)

            question_state = gr.State("")

            start_btn.click(
                fn=next_question,
                inputs=[
                    resume_input, job_desc_input, num_q_input, interview_type,
                    difficulty, company_style, practice_mode, enable_video_recording,
                    question_state, user_answer_audio, user_answer_video
                ],
                outputs=[
                    interviewer_question, user_answer_audio, user_answer_video,
                    start_btn, evaluation_textbox, question_state
                ]
            )

        with gr.TabItem("🎬 Video Playback"):
            gr.Markdown("## Review Your Recorded Interviews")

            refresh_sessions_btn = gr.Button("🔄 Refresh Sessions", variant="primary")
            sessions_table = gr.Markdown("No sessions yet")

            session_dropdown = gr.Dropdown(label="📁 Select Session", choices=[])
            load_session_btn = gr.Button("📂 Load Session")

            session_info = gr.Markdown()

            question_dropdown = gr.Dropdown(label="📝 Select Question", choices=[])
            load_video_btn = gr.Button("▶️ Play Video")

            playback_video = gr.Video(label="📹 Video Playback")
            video_info = gr.Markdown()

            # Event handlers
            refresh_sessions_btn.click(
                fn=list_video_sessions,
                outputs=[sessions_table, session_dropdown]
            )

            load_session_btn.click(
                fn=load_session_for_playback,
                inputs=[session_dropdown],
                outputs=[session_info, playback_video, question_dropdown]
            )

            load_video_btn.click(
                fn=load_specific_video,
                inputs=[session_dropdown, question_dropdown],
                outputs=[playback_video, video_info]
            )

        with gr.TabItem("📊 Analytics"):
            gr.Markdown("## Progress Dashboard")
            refresh_btn = gr.Button("🔄 Refresh")
            dashboard_plot = gr.Plot()
            insights_text = gr.Markdown()

            refresh_btn.click(
                fn=show_analytics_dashboard,
                outputs=[dashboard_plot, insights_text]
            )

if __name__ == "__main__":
    print("🚀 Starting Interview Coach with Video...")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("interview_videos", exist_ok=True)
    print("✅ Ready!")

    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)