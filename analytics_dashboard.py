"""
analytics_dashboard.py - Track interview progress with gorgeous charts
Uses Plotly for interactive, professional visualizations
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd


class InterviewAnalytics:
    def __init__(self, data_file="interview_analytics.json"):
        self.data_file = data_file
        self.data = self._load_data()

    def _load_data(self):
        """Load analytics data from file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {
            'interviews': [],
            'daily_stats': {},
            'skills_progress': {}
        }

    def _save_data(self):
        """Save analytics data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def log_interview(self, interview_data):
        """
        Log a completed interview

        interview_data = {
            'date': datetime,
            'type': str,
            'difficulty': str,
            'scores': dict,
            'speech_metrics': dict,
            'duration': float,
            'questions_count': int
        }
        """
        interview_data['date'] = interview_data.get('date', datetime.now().isoformat())
        self.data['interviews'].append(interview_data)

        # Update daily stats
        date_key = interview_data['date'][:10]  # YYYY-MM-DD
        if date_key not in self.data['daily_stats']:
            self.data['daily_stats'][date_key] = {
                'interviews_count': 0,
                'total_questions': 0,
                'avg_score': 0,
                'total_duration': 0
            }

        stats = self.data['daily_stats'][date_key]
        stats['interviews_count'] += 1
        stats['total_questions'] += interview_data.get('questions_count', 0)
        stats['total_duration'] += interview_data.get('duration', 0)

        # Update skills progress
        scores = interview_data.get('scores', {})
        for skill, score in scores.items():
            if skill not in self.data['skills_progress']:
                self.data['skills_progress'][skill] = []
            self.data['skills_progress'][skill].append({
                'date': interview_data['date'],
                'score': score
            })

        self._save_data()

    def create_progress_dashboard(self):
        """Create comprehensive progress dashboard"""
        if not self.data['interviews']:
            return self._create_empty_dashboard()

        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '📈 Overall Score Trend',
                '🎯 Skills Breakdown',
                '🎤 Speech Quality Over Time',
                '📊 Interview Type Distribution',
                '⏱️ Practice Consistency',
                '🌟 Achievement Progress'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )

        # 1. Overall Score Trend
        dates = [i['date'][:10] for i in self.data['interviews']]
        overall_scores = [i['scores'].get('overall', 0) for i in self.data['interviews']]

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=overall_scores,
                mode='lines+markers',
                name='Overall Score',
                line=dict(color='#00D9FF', width=3),
                marker=dict(size=10, color='#00D9FF'),
                fill='tozeroy',
                fillcolor='rgba(0, 217, 255, 0.1)'
            ),
            row=1, col=1
        )

        # 2. Skills Breakdown (Latest Interview)
        latest = self.data['interviews'][-1]
        skills = latest['scores']
        skill_names = list(skills.keys())
        skill_scores = list(skills.values())

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

        fig.add_trace(
            go.Bar(
                x=skill_scores,
                y=skill_names,
                orientation='h',
                marker=dict(
                    color=colors[:len(skill_names)],
                    line=dict(color='white', width=2)
                ),
                text=skill_scores,
                textposition='outside',
                name='Skills'
            ),
            row=1, col=2
        )

        # 3. Speech Quality Over Time
        filler_counts = [i.get('speech_metrics', {}).get('filler_count', 0)
                         for i in self.data['interviews']]
        wpm = [i.get('speech_metrics', {}).get('words_per_minute', 0)
               for i in self.data['interviews']]

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=filler_counts,
                mode='lines+markers',
                name='Filler Words',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=wpm,
                mode='lines+markers',
                name='Words/Min',
                line=dict(color='#4ECDC4', width=2),
                marker=dict(size=8),
                yaxis='y2'
            ),
            row=2, col=1
        )

        # 4. Interview Type Distribution
        type_counts = defaultdict(int)
        for interview in self.data['interviews']:
            type_counts[interview['type']] += 1

        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']),
                textinfo='label+percent',
                hole=0.4
            ),
            row=2, col=2
        )

        # 5. Practice Consistency (Last 30 days)
        last_30_days = self._get_last_n_days_activity(30)

        fig.add_trace(
            go.Bar(
                x=list(last_30_days.keys()),
                y=list(last_30_days.values()),
                marker=dict(
                    color=list(last_30_days.values()),
                    colorscale='Viridis',
                    showscale=False
                ),
                name='Daily Practice'
            ),
            row=3, col=1
        )

        # 6. Achievement Indicator
        total_interviews = len(self.data['interviews'])
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=avg_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Score", 'font': {'size': 20}},
                delta={'reference': 7, 'increasing': {'color': "#00D9FF"}},
                gauge={
                    'axis': {'range': [None, 10], 'tickwidth': 1},
                    'bar': {'color': "#00D9FF"},
                    'steps': [
                        {'range': [0, 5], 'color': "#FFE5E5"},
                        {'range': [5, 7], 'color': "#FFF9E5"},
                        {'range': [7, 10], 'color': "#E5FFE5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 9
                    }
                }
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title={
                'text': f'🎯 Interview Progress Dashboard - {total_interviews} Interviews Completed',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2C3E50'}
            },
            showlegend=True,
            height=1200,
            template='plotly_white',
            font=dict(family="Arial, sans-serif", size=12),
            paper_bgcolor='#F8F9FA',
            plot_bgcolor='white'
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True, gridcolor='#E8E8E8')
        fig.update_yaxes(title_text="Score (0-10)", row=1, col=1, showgrid=True, gridcolor='#E8E8E8')
        fig.update_xaxes(title_text="Score", row=1, col=2, range=[0, 10])
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Filler Count", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1, tickangle=45)
        fig.update_yaxes(title_text="Interviews", row=3, col=1)

        return fig

    def create_skills_heatmap(self):
        """Create heatmap showing skills improvement over time"""
        if not self.data['skills_progress']:
            return None

        # Prepare data
        skills = list(self.data['skills_progress'].keys())
        dates = sorted(list(set([
            entry['date'][:10]
            for skill_data in self.data['skills_progress'].values()
            for entry in skill_data
        ])))

        # Create matrix
        matrix = []
        for skill in skills:
            skill_row = []
            skill_data = {entry['date'][:10]: entry['score']
                          for entry in self.data['skills_progress'][skill]}
            for date in dates:
                skill_row.append(skill_data.get(date, None))
            matrix.append(skill_row)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=dates,
            y=skills,
            colorscale='RdYlGn',
            text=matrix,
            texttemplate='%{text:.1f}',
            textfont={"size": 12},
            colorbar=dict(title="Score"),
            hoverongaps=False
        ))

        fig.update_layout(
            title='🔥 Skills Progress Heatmap',
            xaxis_title='Date',
            yaxis_title='Skill',
            height=400,
            template='plotly_white'
        )

        return fig

    def create_comparison_chart(self):
        """Compare current performance vs first interview"""
        if len(self.data['interviews']) < 2:
            return None

        first = self.data['interviews'][0]
        latest = self.data['interviews'][-1]

        categories = list(first['scores'].keys())
        first_scores = [first['scores'][cat] for cat in categories]
        latest_scores = [latest['scores'][cat] for cat in categories]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=first_scores,
            theta=categories,
            fill='toself',
            name='First Interview',
            line=dict(color='#FF6B6B', width=2)
        ))

        fig.add_trace(go.Scatterpolar(
            r=latest_scores,
            theta=categories,
            fill='toself',
            name='Latest Interview',
            line=dict(color='#00D9FF', width=2)
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10])
            ),
            showlegend=True,
            title='📊 First vs Latest Interview Comparison',
            height=500,
            template='plotly_white'
        )

        return fig

    def get_insights(self):
        """Generate text insights from data"""
        if not self.data['interviews']:
            return "No interviews completed yet. Start practicing to see insights!"

        total = len(self.data['interviews'])
        latest = self.data['interviews'][-1]
        avg_score = sum([i['scores'].get('overall', 0) for i in self.data['interviews']]) / total

        # Find strongest skill
        latest_scores = latest['scores']
        strongest_skill = max(latest_scores.items(), key=lambda x: x[1])
        weakest_skill = min(latest_scores.items(), key=lambda x: x[1])

        # Calculate improvement
        if len(self.data['interviews']) > 1:
            first_score = self.data['interviews'][0]['scores'].get('overall', 0)
            improvement = latest['scores'].get('overall', 0) - first_score
            improvement_text = f"📈 {improvement:+.1f} points since first interview"
        else:
            improvement_text = "Complete more interviews to see improvement!"

        insights = f"""
### 🎯 Your Interview Insights

**Total Interviews Completed:** {total}
**Average Score:** {avg_score:.1f}/10
**Latest Score:** {latest['scores'].get('overall', 0):.1f}/10

**Strongest Skill:** {strongest_skill[0].replace('_', ' ').title()} ({strongest_skill[1]:.1f}/10) 💪
**Area to Improve:** {weakest_skill[0].replace('_', ' ').title()} ({weakest_skill[1]:.1f}/10) 📚

**Progress:** {improvement_text}

**Practice Consistency:** {self._get_consistency_message()}

**Speech Quality:** {self._get_speech_insight(latest)}
"""
        return insights

    def _get_last_n_days_activity(self, n=30):
        """Get interview count for last N days"""
        activity = defaultdict(int)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n)

        for interview in self.data['interviews']:
            date = datetime.fromisoformat(interview['date'][:10])
            if start_date <= date <= end_date:
                activity[interview['date'][:10]] += 1

        # Fill in missing days with 0
        current = start_date
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            if date_str not in activity:
                activity[date_str] = 0
            current += timedelta(days=1)

        return dict(sorted(activity.items()))

    def _get_consistency_message(self):
        """Get message about practice consistency"""
        if len(self.data['interviews']) < 2:
            return "Just getting started!"

        dates = [datetime.fromisoformat(i['date'][:10]) for i in self.data['interviews']]
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg_gap = sum(gaps) / len(gaps)

        if avg_gap <= 1:
            return "🔥 Practicing daily! Amazing consistency!"
        elif avg_gap <= 3:
            return "💪 Great consistency! Keep it up!"
        elif avg_gap <= 7:
            return "👍 Good practice rhythm."
        else:
            return "📅 Try to practice more regularly for better results."

    def _get_speech_insight(self, interview):
        """Get insight about speech quality"""
        metrics = interview.get('speech_metrics', {})
        filler_count = metrics.get('filler_count', 0)
        wpm = metrics.get('words_per_minute', 0)

        insights = []

        if filler_count < 3:
            insights.append("✨ Excellent - minimal filler words!")
        elif filler_count < 7:
            insights.append("👍 Good - acceptable filler word usage")
        else:
            insights.append("📝 Work on reducing filler words (um, uh, like)")

        if 120 <= wpm <= 160:
            insights.append("Perfect speaking pace!")
        elif wpm < 120:
            insights.append("Speak slightly faster for more energy")
        else:
            insights.append("Slow down a bit for clarity")

        return " | ".join(insights)

    def _create_empty_dashboard(self):
        """Create placeholder dashboard when no data"""
        fig = go.Figure()
        fig.add_annotation(
            text="📊 No interviews yet!<br>Complete your first interview to see analytics.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#7F8C8D")
        )
        fig.update_layout(
            height=600,
            template='plotly_white',
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig


# Example usage
def demo_analytics():
    """Demo the analytics dashboard"""
    analytics = InterviewAnalytics()

    # Simulate some interview data
    for i in range(5):
        interview_data = {
            'date': (datetime.now() - timedelta(days=i * 2)).isoformat(),
            'type': 'technical',
            'difficulty': 'mid',
            'scores': {
                'technical_skills': 6 + i * 0.5,
                'communication': 7 + i * 0.3,
                'problem_solving': 6.5 + i * 0.4,
                'cultural_fit': 7.5 + i * 0.2,
                'experience_relevance': 7 + i * 0.3,
                'overall': 7 + i * 0.3
            },
            'speech_metrics': {
                'filler_count': max(0, 10 - i * 2),
                'words_per_minute': 140 + i * 5
            },
            'duration': 1800,
            'questions_count': 5
        }
        analytics.log_interview(interview_data)

    # Create dashboard
    fig = analytics.create_progress_dashboard()
    fig.write_html('dashboard_demo.html')
    print("✅ Dashboard saved to dashboard_demo.html")

    # Print insights
    print(analytics.get_insights())


if __name__ == "__main__":
    demo_analytics()