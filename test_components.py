"""
test_components.py - Quick test of all components
Run this to verify everything works before starting the app
"""

import os
import sys

print("🧪 Testing Interview Coach Components...\n")

# Test 1: Imports
print("1️⃣ Testing imports...")
try:
    from groq import Groq

    print("   ✅ Groq")
except Exception as e:
    print(f"   ❌ Groq: {e}")
    sys.exit(1)

try:
    from gtts import gTTS

    print("   ✅ gTTS")
except Exception as e:
    print(f"   ❌ gTTS: {e}")

try:
    from faster_whisper import WhisperModel

    print("   ✅ Faster Whisper")
except Exception as e:
    print(f"   ❌ Faster Whisper: {e}")

try:
    import gradio as gr

    print("   ✅ Gradio")
except Exception as e:
    print(f"   ❌ Gradio: {e}")
    sys.exit(1)

try:
    from job_scraper import JobScraper

    print("   ✅ Job Scraper")
except Exception as e:
    print(f"   ❌ Job Scraper: {e}")
    sys.exit(1)

try:
    from rag_system import get_rag_system

    print("   ✅ RAG System")
except Exception as e:
    print(f"   ❌ RAG System: {e}")
    sys.exit(1)

try:
    from analytics_dashboard import InterviewAnalytics

    print("   ✅ Analytics")
except Exception as e:
    print(f"   ❌ Analytics: {e}")

print("\n2️⃣ Testing TTS (Text-to-Speech)...")
try:
    tts = gTTS(text="Testing audio system", lang='en')
    test_file = "test_audio.mp3"
    tts.save(test_file)

    if os.path.exists(test_file):
        size = os.path.getsize(test_file)
        print(f"   ✅ TTS works! Generated {size} bytes")
        os.remove(test_file)
    else:
        print("   ❌ TTS file not created")
except Exception as e:
    print(f"   ❌ TTS failed: {e}")

print("\n3️⃣ Testing RAG System...")
try:
    rag = get_rag_system()

    # Test query
    questions = rag.get_relevant_questions(
        resume_text="Python developer with ML experience",
        job_description="Looking for data scientist",
        interview_type="technical",
        difficulty="mid",
        company_style="general",
        num_questions=3
    )

    if len(questions) > 0:
        print(f"   ✅ RAG works! Retrieved {len(questions)} questions")
        print(f"   📝 Sample: {questions[0]['question'][:60]}...")
    else:
        print("   ⚠️ RAG returned 0 questions")

except Exception as e:
    print(f"   ❌ RAG failed: {e}")
    import traceback

    traceback.print_exc()

print("\n4️⃣ Testing Job Scraper...")
try:
    scraper = JobScraper()

    # Test with a generic URL (will likely fail but shouldn't crash)
    result = scraper.scrape_job("https://example.com")

    if 'title' in result:
        print(f"   ✅ Scraper works! (returned fallback)")
    else:
        print(f"   ⚠️ Scraper returned unexpected format")

except Exception as e:
    print(f"   ❌ Scraper failed: {e}")

print("\n5️⃣ Testing Analytics...")
try:
    analytics = InterviewAnalytics()

    # Test logging
    analytics.log_interview({
        'date': '2024-01-01T12:00:00',
        'type': 'technical',
        'difficulty': 'mid',
        'scores': {'overall': 8, 'technical_skills': 8},
        'speech_metrics': {'filler_count': 5},
        'duration': 1800,
        'questions_count': 5
    })

    print("   ✅ Analytics works!")

except Exception as e:
    print(f"   ❌ Analytics failed: {e}")

print("\n6️⃣ Checking directories...")
dirs_to_check = ['checkpoints', 'chroma_db', 'interview_history']
for dir_name in dirs_to_check:
    if os.path.exists(dir_name):
        print(f"   ✅ {dir_name}/ exists")
    else:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   📁 Created {dir_name}/")

print("\n" + "=" * 50)
print("✅ COMPONENT TEST COMPLETE!")
print("=" * 50)
print("\n💡 Next steps:")
print("   1. Make sure GROQ_API_KEY is set in myapp.py")
print("   2. Run: python myapp.py")
print("   3. Open the URL shown in terminal")
print("\n🚀 Ready to launch!")