"""
video_handler.py - Video recording, storage, and playback system
Handles interview video with timestamps for each question
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import tempfile


class VideoInterviewManager:
    def __init__(self, storage_dir="interview_videos"):
        """Initialize video manager"""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # Current session data
        self.current_session_id = None
        self.current_session_data = None
        self.video_segments = []

    def start_new_session(self, resume_name, job_title, interview_type):
        """Start a new interview recording session"""
        # Generate unique session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = f"interview_{timestamp}"

        # Create session directory
        session_dir = os.path.join(self.storage_dir, self.current_session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Initialize session data
        self.current_session_data = {
            "session_id": self.current_session_id,
            "start_time": datetime.now().isoformat(),
            "resume_name": resume_name,
            "job_title": job_title,
            "interview_type": interview_type,
            "video_segments": [],
            "qa_pairs": [],
            "total_duration": 0
        }

        self.video_segments = []

        print(f"📹 Started video session: {self.current_session_id}")
        return self.current_session_id

    def save_video_segment(self, video_file_path, question_number, question_text,
                           answer_transcript="", duration=0):
        """Save a video segment for a specific question"""
        if not self.current_session_id:
            print("⚠️ No active session")
            return None

        if not video_file_path or not os.path.exists(video_file_path):
            print("⚠️ Video file not found")
            return None

        try:
            # Create filename
            segment_filename = f"q{question_number}_{datetime.now().strftime('%H%M%S')}.mp4"
            session_dir = os.path.join(self.storage_dir, self.current_session_id)
            segment_path = os.path.join(session_dir, segment_filename)

            # Copy video file to session directory
            shutil.copy2(video_file_path, segment_path)

            # Get file size
            file_size = os.path.getsize(segment_path)

            # Create segment metadata
            segment_data = {
                "question_number": question_number,
                "question_text": question_text,
                "answer_transcript": answer_transcript,
                "video_filename": segment_filename,
                "video_path": segment_path,
                "file_size": file_size,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }

            # Add to current session
            self.video_segments.append(segment_data)
            self.current_session_data["video_segments"].append(segment_data)

            print(f"✅ Saved video segment {question_number}: {file_size} bytes")
            return segment_path

        except Exception as e:
            print(f"❌ Error saving video segment: {str(e)}")
            return None

    def add_qa_data(self, question_number, question_text, answer_transcript,
                    speech_analysis, scores, feedback):
        """Add Q&A data to current session"""
        if not self.current_session_id:
            return

        qa_data = {
            "question_number": question_number,
            "question": question_text,
            "answer": answer_transcript,
            "speech_analysis": speech_analysis,
            "scores": scores,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }

        self.current_session_data["qa_pairs"].append(qa_data)

    def finalize_session(self, final_scores, final_evaluation):
        """Finalize and save session data"""
        if not self.current_session_id:
            return None

        # Add final data
        self.current_session_data["end_time"] = datetime.now().isoformat()
        self.current_session_data["final_scores"] = final_scores
        self.current_session_data["final_evaluation"] = final_evaluation

        # Calculate total duration
        total_duration = sum([seg.get("duration", 0) for seg in self.video_segments])
        self.current_session_data["total_duration"] = total_duration

        # Save session metadata as JSON
        session_dir = os.path.join(self.storage_dir, self.current_session_id)
        metadata_path = os.path.join(session_dir, "session_metadata.json")

        with open(metadata_path, 'w') as f:
            json.dump(self.current_session_data, f, indent=2)

        print(f"✅ Finalized session: {self.current_session_id}")
        print(f"   Videos: {len(self.video_segments)}")
        print(f"   Duration: {total_duration:.1f}s")

        session_id = self.current_session_id

        # Reset for next session
        self.current_session_id = None
        self.current_session_data = None
        self.video_segments = []

        return session_id

    def get_all_sessions(self):
        """Get list of all recorded sessions"""
        sessions = []

        if not os.path.exists(self.storage_dir):
            return sessions

        for session_id in os.listdir(self.storage_dir):
            session_dir = os.path.join(self.storage_dir, session_id)
            if not os.path.isdir(session_dir):
                continue

            metadata_path = os.path.join(session_dir, "session_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        session_data = json.load(f)

                    # Add summary info
                    sessions.append({
                        "session_id": session_id,
                        "start_time": session_data.get("start_time"),
                        "job_title": session_data.get("job_title", "Unknown"),
                        "interview_type": session_data.get("interview_type", "Unknown"),
                        "num_videos": len(session_data.get("video_segments", [])),
                        "total_duration": session_data.get("total_duration", 0),
                        "final_score": session_data.get("final_scores", {}).get("overall", "N/A")
                    })
                except Exception as e:
                    print(f"⚠️ Error reading session {session_id}: {e}")

        # Sort by date (newest first)
        sessions.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        return sessions

    def load_session(self, session_id):
        """Load a specific session"""
        session_dir = os.path.join(self.storage_dir, session_id)
        metadata_path = os.path.join(session_dir, "session_metadata.json")

        if not os.path.exists(metadata_path):
            print(f"❌ Session not found: {session_id}")
            return None

        try:
            with open(metadata_path, 'r') as f:
                session_data = json.load(f)

            print(f"✅ Loaded session: {session_id}")
            return session_data
        except Exception as e:
            print(f"❌ Error loading session: {e}")
            return None

    def get_video_segment_path(self, session_id, question_number):
        """Get path to specific video segment"""
        session_data = self.load_session(session_id)
        if not session_data:
            return None

        for segment in session_data.get("video_segments", []):
            if segment.get("question_number") == question_number:
                return segment.get("video_path")

        return None

    def export_session_summary(self, session_id):
        """Export session summary as markdown"""
        session_data = self.load_session(session_id)
        if not session_data:
            return None

        # Create markdown summary
        summary = f"""# Interview Recording Summary

## Session Details
- **Session ID:** {session_id}
- **Date:** {session_data.get('start_time', 'Unknown')[:10]}
- **Job Title:** {session_data.get('job_title', 'Unknown')}
- **Interview Type:** {session_data.get('interview_type', 'Unknown')}
- **Total Duration:** {session_data.get('total_duration', 0):.1f} seconds

## Final Scores
"""

        final_scores = session_data.get('final_scores', {})
        for skill, score in final_scores.items():
            summary += f"- **{skill.replace('_', ' ').title()}:** {score}/10\n"

        summary += "\n## Questions & Answers\n"

        for qa in session_data.get('qa_pairs', []):
            summary += f"\n### Q{qa['question_number']}: {qa['question'][:80]}...\n"
            summary += f"**Answer:** {qa['answer'][:200]}...\n\n"

            speech = qa.get('speech_analysis', {})
            if speech:
                summary += f"- Words per minute: {speech.get('words_per_minute', 'N/A')}\n"
                summary += f"- Filler words: {speech.get('filler_count', 'N/A')}\n"

        summary += f"\n## Final Evaluation\n{session_data.get('final_evaluation', 'N/A')}\n"

        # Save to file
        session_dir = os.path.join(self.storage_dir, session_id)
        summary_path = os.path.join(session_dir, "summary.md")

        with open(summary_path, 'w') as f:
            f.write(summary)

        print(f"✅ Exported summary: {summary_path}")
        return summary_path

    def delete_session(self, session_id):
        """Delete a session and all its videos"""
        session_dir = os.path.join(self.storage_dir, session_id)

        if os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir)
                print(f"🗑️ Deleted session: {session_id}")
                return True
            except Exception as e:
                print(f"❌ Error deleting session: {e}")
                return False

        return False

    def get_storage_stats(self):
        """Get storage statistics"""
        total_size = 0
        total_videos = 0
        total_sessions = 0

        if not os.path.exists(self.storage_dir):
            return {"total_size": 0, "total_videos": 0, "total_sessions": 0}

        for session_id in os.listdir(self.storage_dir):
            session_dir = os.path.join(self.storage_dir, session_id)
            if not os.path.isdir(session_dir):
                continue

            total_sessions += 1

            for file in os.listdir(session_dir):
                file_path = os.path.join(session_dir, file)
                if os.path.isfile(file_path) and file.endswith('.mp4'):
                    total_size += os.path.getsize(file_path)
                    total_videos += 1

        return {
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_videos": total_videos,
            "total_sessions": total_sessions
        }


# Test function
if __name__ == "__main__":
    print("🧪 Testing Video Manager...")

    manager = VideoInterviewManager()

    # Start session
    session_id = manager.start_new_session(
        resume_name="John Doe",
        job_title="Data Scientist",
        interview_type="technical"
    )

    print(f"Session ID: {session_id}")

    # Get all sessions
    sessions = manager.get_all_sessions()
    print(f"Total sessions: {len(sessions)}")

    # Get stats
    stats = manager.get_storage_stats()
    print(f"Storage stats: {stats}")

    print("\n✅ Video manager test complete!")