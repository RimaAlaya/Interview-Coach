# 🎯 Advanced AI Interview Coach

> An intelligent, multi-agent interview coaching system with real-time analysis, emotion detection, and company-specific training

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-5.49.1-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![Interview Coach Demo](demo.gif)

## 🌟 Features

### 🤖 Multi-Agent AI Architecture
- **5 Specialized AI Agents** working in harmony:
  - 📄 Resume Analyst - Extracts key skills and experiences
  - 💼 Job Expert - Analyzes job requirements
  - 🎯 Interview Strategist - Plans question flow
  - 🎤 Interviewer - Asks contextual questions
  - 📊 Evaluator - Provides comprehensive feedback

### 🎭 Multiple Interview Types
- **Behavioral**: STAR method questions about past experiences
- **Technical**: Problem-solving and technical competency
- **Leadership**: Management and decision-making scenarios
- **Case Study**: Business analysis and frameworks

### 📈 Adaptive Difficulty Levels
- **Junior** (0-2 years): Entry-level foundational questions
- **Mid** (3-5 years): Intermediate practical scenarios
- **Senior** (5+ years): Advanced strategic thinking

### 🏢 Company-Specific Training
Tailored interview styles for:
- 🔍 **Google**: Algorithmic thinking, innovation, scalability
- 📦 **Amazon**: Leadership principles, customer obsession
- 💻 **Microsoft**: Collaboration, growth mindset
- 👥 **Meta**: Impact-driven, moving fast
- 📊 **Consulting**: Case studies, frameworks
- 🚀 **Startup**: Versatility, ownership, scrappiness

### 🎤 Advanced Speech Analysis
Real-time analysis of your delivery:
- ⏱️ **Speaking Pace**: Words per minute tracking
- 🗣️ **Filler Words**: Detection of um, uh, like, etc.
- 📏 **Answer Length**: Optimal response duration
- ⏸️ **Pauses**: Natural vs. awkward silence detection

### 😊 Emotion & Sentiment Analysis
Powered by Hugging Face Transformers:
- Confidence level detection
- Emotion recognition (joy, fear, neutral, etc.)
- Sentiment analysis of responses
- Delivery coaching tips

### 📊 Comprehensive Scoring System
Multi-dimensional evaluation (0-10 scale):
- 🔧 **Technical Skills**: Domain knowledge
- 💬 **Communication**: Clarity and articulation
- 🧩 **Problem Solving**: Analytical thinking
- 🤝 **Cultural Fit**: Values alignment
- 📚 **Experience Relevance**: Role match
- ⭐ **Overall Score**: Weighted average

### 🎓 Practice vs. Real Mode
- **Practice Mode**: 
  - ✅ Hints with questions
  - ✅ Real-time feedback
  - ✅ Speech analysis
  - ✅ Unlimited retries
  
- **Real Mode**: 
  - 🎯 Realistic pressure simulation
  - 🎯 No hints or real-time feedback
  - 🎯 Final evaluation only

### 💾 Save/Resume Functionality
- ⏸️ Pause interviews anytime
- 📁 Save progress as checkpoints
- ▶️ Resume from where you left off
- 📄 Export complete interview history

### 📹 Multi-Modal Input
- 🎤 Audio recording via microphone
- 📹 Video recording (optional, for better audio)
- 📄 PDF resume upload
- 📝 Job description input

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Microphone access
- (Optional) Webcam for video mode

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/interview-coach-advanced.git
cd interview-coach-advanced
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Get Groq API Key** (FREE!)
   - Go to [console.groq.com](https://console.groq.com/)
   - Sign up (no credit card required)
   - Create API key
   - Copy your key

5. **Configure API Key**
   - Open `myapp.py`
   - Replace `YOUR_GROQ_API_KEY_HERE` with your actual key:
   ```python
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```

6. **Run the application**
```bash
python myapp.py
```

7. **Open in browser**
   - Gradio will automatically open your browser
   - Or visit the URL shown in terminal (usually `http://localhost:7860`)

## 📖 Usage Guide

### Basic Interview Flow

1. **Upload Materials**
   - Upload your resume (PDF format)
   - Paste the job description you're applying for

2. **Configure Settings**
   - Select interview type (Behavioral, Technical, etc.)
   - Choose difficulty level (Junior, Mid, Senior)
   - Pick company style (Google, Amazon, etc.)
   - Toggle Practice Mode (recommended for learning)
   - Set number of questions (1-10)

3. **Start Interview**
   - Click "Start Interview"
   - Listen to the question (plays automatically)
   - Record your answer via microphone
   - Click "Submit Answer"

4. **Review Feedback**
   - Practice Mode: Get instant feedback after each answer
   - Real Mode: Get comprehensive evaluation at the end
   - See detailed scores and speech analysis

5. **Save Progress** (Optional)
   - Go to "Save/Load" tab
   - Click "Save Current Interview"
   - Resume later by uploading checkpoint file

### Tips for Best Results

#### 🎤 Recording Quality
- Use a quiet environment
- Speak clearly and at normal pace (120-160 words/minute)
- Position microphone 6-12 inches from mouth
- Test audio before starting

#### 💡 Answer Structure (STAR Method)
- **S**ituation: Set the context
- **T**ask: Explain your responsibility
- **A**ction: Describe what you did
- **R**esult: Share the outcome with metrics

#### ⏱️ Answer Length
- Aim for 1-2 minutes per answer
- 150-300 words is optimal
- Be specific but concise

#### 🗣️ Delivery Tips
- Minimize filler words (um, uh, like)
- Use concrete examples with numbers
- Show enthusiasm through tone
- Maintain steady pace

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────┐
│                  User Interface                  │
│              (Gradio Web App)                   │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐       ┌──────────────┐
│  PDF Reader  │       │  Audio I/O   │
│  (PyPDF2)    │       │ (gTTS/Whisper)│
└──────┬───────┘       └──────┬───────┘
       │                      │
       └──────────┬───────────┘
                  │
         ┌────────▼─────────┐
         │   5 AI Agents    │
         │  (Groq LLM API)  │
         └────────┬─────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌─────────┐
│Resume  │  │Interview │  │Evaluator│
│Analyst │  │Strategist│  │         │
└────────┘  └──────────┘  └─────────┘
```

### Agent Workflow

```
1. Resume Analyst → Analyzes candidate background
2. Job Expert → Identifies required skills
3. Interview Strategist → Plans question flow
4. Interviewer → Asks contextual questions
5. Evaluator → Provides comprehensive feedback
```

## 📊 Scoring Methodology

### Score Calculation
Each dimension is scored 0-10 based on:

**Technical Skills**
- Depth of knowledge demonstrated
- Accuracy of technical concepts
- Real-world application examples

**Communication**
- Clarity of expression
- Structured answers (STAR method)
- Active listening indicators

**Problem Solving**
- Analytical approach
- Creativity in solutions
- Handling of edge cases

**Cultural Fit**
- Alignment with stated values
- Team collaboration examples
- Growth mindset indicators

**Experience Relevance**
- Direct skill matches
- Transferable skills
- Career progression logic

## 📁 Project Structure

```
interview-coach-advanced/
├── myapp.py                    # Main application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── checkpoints/                # Saved interview progress
│   └── checkpoint_*.json
├── interview_history/          # Completed interviews
│   └── interview_*.json
├── examples/                   # Example files
│   ├── example_resume.pdf
│   └── example_job_description.txt
└── docs/
    ├── CONTRIBUTING.md
    └── API_REFERENCE.md
```

## 🔧 Configuration

### Environment Variables (Optional)
Create `.env` file:
```env
GROQ_API_KEY=your_key_here
WHISPER_MODEL_SIZE=base  # tiny, base, small, medium, large
```

### Customization Options

#### Add Custom Company Style
Edit `COMPANY_STYLES` in `myapp.py`:
```python
"your_company": {
    "description": "Your company interview style",
    "focus": "Your company's focus areas and values"
}
```

#### Adjust Speech Analysis Thresholds
```python
# In analyze_speech_quality()
if words_per_minute < 100:  # Adjust threshold
    pace = "too slow"
```

## 🐛 Troubleshooting

### Common Issues

**1. Microphone not working**
- Grant browser microphone permissions
- Check system audio settings
- Try different browser (Chrome recommended)

**2. Groq API errors**
```
Error: API key invalid
```
Solution: Verify your API key is correct and active

**3. Slow transcription**
```python
# Use smaller Whisper model in myapp.py
def transcribe_audio_faster_whisper(audio_file_path, model_size="tiny")
```

**4. Out of memory**
- Use smaller Whisper model (tiny or base)
- Close other applications
- Restart Python kernel

**5. Emotion analysis not loading**
```bash
# Manually install transformers dependencies
pip install torch transformers sentencepiece
```

## 🚀 Advanced Features

### Export Interview Data

Interviews are saved as JSON:
```json
{
  "timestamp": "2024-12-16T10:30:00",
  "candidate": "John Doe",
  "position": "Senior Software Engineer",
  "questions_answers": {...},
  "scores": {...},
  "evaluation": "..."
}
```

### Batch Processing
Process multiple candidates:
```python
# Coming soon: CLI mode for batch interviews
python myapp.py --batch candidates.csv
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md)

### Development Setup
```bash
git clone https://github.com/yourusername/interview-coach-advanced.git
cd interview-coach-advanced
pip install -r requirements-dev.txt
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

## 📝 Roadmap

- [ ] Mobile app (React Native)
- [ ] Multi-language support
- [ ] Real-time avatar interviewer
- [ ] Integration with ATS systems
- [ ] Team collaboration features
- [ ] Analytics dashboard
- [ ] Browser extension

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- **Groq** for fast LLM inference
- **Meta** for Llama 3 model
- **OpenAI** for Whisper model
- **Hugging Face** for emotion models
- **Gradio** for UI framework

## 📧 Contact

**Your Name**
- LinkedIn: [your-profile](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## ⭐ Star History

If this project helped you, please give it a star! ⭐

## 📊 Stats

- **5** AI Agents
- **4** Interview Types
- **7** Company Styles
- **3** Difficulty Levels
- **6** Scoring Dimensions
- **100%** Free & Open Source

---

**Built with ❤️ using AI | Powered by Groq 🚀**