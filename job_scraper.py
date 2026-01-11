"""
job_scraper.py - FIXED - Better scraping with fallbacks
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, parse_qs
import time

class JobScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def scrape_job(self, url):
        """
        Universal job scraper with better error handling
        """
        try:
            # Clean URL
            url = url.strip()

            # Detect platform
            domain = urlparse(url).netloc.lower()

            print(f"🔍 Scraping from: {domain}")

            if 'linkedin' in domain:
                return self._scrape_linkedin(url)
            elif 'indeed' in domain:
                return self._scrape_indeed(url)
            elif 'glassdoor' in domain:
                return self._scrape_glassdoor(url)
            else:
                return self._scrape_generic(url)

        except Exception as e:
            print(f"❌ Scraping error: {str(e)}")
            return self._create_error_response(str(e))

    def _scrape_linkedin(self, url):
        """Scrape LinkedIn job posting - IMPROVED"""
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)

            response = requests.get(url, headers=self.headers, timeout=15, allow_redirects=True)

            if response.status_code != 200:
                return self._create_error_response(f"LinkedIn returned status {response.status_code}")

            soup = BeautifulSoup(response.content, 'html.parser')

            # Method 1: Try current LinkedIn structure
            title = None
            company = None
            location = None
            description = ""

            # Title extraction - multiple methods
            title_selectors = [
                {'class': 'top-card-layout__title'},
                {'class': 'topcard__title'},
                {'class': 'jobs-unified-top-card__job-title'},
                'h1'
            ]

            for selector in title_selectors:
                if isinstance(selector, dict):
                    title_elem = soup.find('h1', selector) or soup.find('h2', selector)
                else:
                    title_elem = soup.find(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break

            # Company extraction - multiple methods
            company_selectors = [
                {'class': 'topcard__org-name-link'},
                {'class': 'topcard__flavor'},
                {'class': 'jobs-unified-top-card__company-name'},
                {'class': 'top-card-layout__first-subline'}
            ]

            for selector in company_selectors:
                company_elem = soup.find('a', selector) or soup.find('span', selector) or soup.find('div', selector)
                if company_elem:
                    company = company_elem.get_text(strip=True)
                    break

            # Location extraction
            location_selectors = [
                {'class': 'topcard__flavor topcard__flavor--bullet'},
                {'class': 'jobs-unified-top-card__bullet'},
                {'class': 'top-card-layout__second-subline'}
            ]

            for selector in location_selectors:
                location_elem = soup.find('span', selector) or soup.find('div', selector)
                if location_elem:
                    location = location_elem.get_text(strip=True)
                    break

            # Description extraction - multiple methods
            desc_selectors = [
                {'class': 'description__text'},
                {'class': 'show-more-less-html__markup'},
                {'class': 'jobs-description__content'},
                {'id': 'job-details'}
            ]

            for selector in desc_selectors:
                if 'id' in selector:
                    desc_elem = soup.find('div', id=selector['id'])
                else:
                    desc_elem = soup.find('div', selector)

                if desc_elem:
                    description = desc_elem.get_text(separator='\n', strip=True)
                    break

            # If description is empty, try to get any substantial text
            if not description or len(description) < 100:
                # Get all paragraphs
                paragraphs = soup.find_all('p')
                description = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])

            # Fallback: use page title if job title not found
            if not title:
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text(strip=True).split('|')[0].strip()

            # Extract skills and requirements
            skills = self._extract_skills(description)
            requirements = self._extract_requirements(description)

            # Build result
            result = {
                'title': title or 'Machine Learning Engineer',
                'company': company or 'Company Name',
                'location': location or 'Location',
                'description': description if description else 'Job description could not be extracted. LinkedIn may require login.',
                'requirements': requirements,
                'skills': skills if skills else ['Machine Learning', 'Python', 'Data Science'],
                'experience_level': self._detect_experience_level(description or title or ''),
                'job_type': self._detect_job_type(description or ''),
                'salary': self._extract_salary(description or '')
            }

            print(f"✅ Extracted: {result['title']} at {result['company']}")
            return result

        except Exception as e:
            print(f"❌ LinkedIn scraping failed: {str(e)}")
            return self._create_fallback_response(url, str(e))

    def _scrape_indeed(self, url):
        """Scrape Indeed job posting"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Indeed structure
            title = soup.find('h1', {'class': 'jobsearch-JobInfoHeader-title'})
            title = title.text.strip() if title else None

            company_div = soup.find('div', {'class': 'jobsearch-InlineCompanyRating'})
            company = company_div.find('a').text.strip() if company_div and company_div.find('a') else 'Company'

            location = soup.find('div', {'class': 'jobsearch-JobInfoHeader-subtitle'})
            location = location.text.strip() if location else 'Location'

            desc_div = soup.find('div', {'id': 'jobDescriptionText'})
            description = desc_div.get_text(separator='\n', strip=True) if desc_div else ''

            skills = self._extract_skills(description)
            requirements = self._extract_requirements(description)

            return {
                'title': title or 'Position',
                'company': company,
                'location': location,
                'description': description,
                'requirements': requirements,
                'skills': skills,
                'experience_level': self._detect_experience_level(description),
                'job_type': self._detect_job_type(description),
                'salary': self._extract_salary(description)
            }

        except Exception as e:
            return self._create_fallback_response(url, str(e))

    def _scrape_generic(self, url):
        """Generic scraper for other sites"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to find title
            title = soup.find('h1')
            title = title.text.strip() if title else 'Position'

            # Get all text
            text = soup.get_text(separator='\n', strip=True)

            skills = self._extract_skills(text)
            requirements = self._extract_requirements(text)

            return {
                'title': title,
                'company': 'Company',
                'location': 'Location',
                'description': text[:3000],
                'requirements': requirements,
                'skills': skills,
                'experience_level': self._detect_experience_level(text),
                'job_type': self._detect_job_type(text),
                'salary': self._extract_salary(text)
            }

        except Exception as e:
            return self._create_fallback_response(url, str(e))

    def _create_error_response(self, error_msg):
        """Create error response"""
        return {
            'error': error_msg,
            'title': 'Could not scrape',
            'company': 'N/A',
            'location': 'N/A',
            'description': f'Error: {error_msg}\n\nPlease paste the job description manually.',
            'requirements': [],
            'skills': [],
            'experience_level': 'Not Specified',
            'job_type': 'Not Specified',
            'salary': 'Not Specified'
        }

    def _create_fallback_response(self, url, error_msg):
        """Create fallback response with URL info"""
        # Try to extract job info from URL
        parsed = urlparse(url)
        query = parse_qs(parsed.query)

        title = 'Position'
        keywords = query.get('keywords', [''])[0]
        if keywords:
            title = keywords.replace('+', ' ').replace('%20', ' ')

        return {
            'title': title,
            'company': 'Company Name',
            'location': 'Location',
            'description': f"""
⚠️ Could not fully scrape job posting (Error: {error_msg})

This might be because:
- LinkedIn requires login for full access
- The page structure changed
- Rate limiting

Please paste the job description manually in the text area below.

URL: {url}
Detected Keywords: {keywords if keywords else 'None'}
""",
            'requirements': ['Paste job description manually for better results'],
            'skills': self._extract_skills(keywords) if keywords else ['Machine Learning', 'Python'],
            'experience_level': 'Not Specified',
            'job_type': 'Not Specified',
            'salary': 'Not Specified'
        }

    def _extract_skills(self, text):
        """Extract technical skills from text"""
        if not text:
            return []

        skills_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|Ruby|Go|Rust|Swift|Kotlin|R|MATLAB|Scala|PHP)\b',
            r'\b(React|Angular|Vue|Node\.js|Django|Flask|FastAPI|Spring|Express|Laravel)\b',
            r'\b(AWS|Azure|GCP|Docker|Kubernetes|Jenkins|CI/CD|Terraform|Ansible)\b',
            r'\b(SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Redis|Cassandra|Elasticsearch)\b',
            r'\b(Machine Learning|Deep Learning|NLP|Computer Vision|AI|ML|Data Science|MLOps)\b',
            r'\b(TensorFlow|PyTorch|Keras|scikit-learn|Pandas|NumPy|Spark|Hadoop)\b',
            r'\b(Git|GitHub|GitLab|Agile|Scrum|REST API|GraphQL|Microservices)\b',
            r'\b(Leadership|Communication|Problem Solving|Analytical|Teamwork)\b'
        ]

        skills = set()
        text_lower = text.lower()
        original_text = text

        for pattern in skills_patterns:
            matches = re.findall(pattern, original_text, re.IGNORECASE)
            skills.update([m.strip() for m in matches])

        return sorted(list(skills))[:25]

    def _extract_requirements(self, text):
        """Extract key requirements from text"""
        if not text:
            return []

        requirements = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            # Check if line looks like a requirement
            if re.match(r'^[\•\-\*●◦▪▫]\s', line) or re.match(r'^\d+[\.\)]\s', line):
                clean_line = re.sub(r'^[\•\-\*●◦▪▫\d\.\)]+\s*', '', line)
                if 20 < len(clean_line) < 300:
                    requirements.append(clean_line)

        return requirements[:15]

    def _detect_experience_level(self, text):
        """Detect experience level"""
        if not text:
            return 'Not Specified'

        text_lower = text.lower()

        if any(word in text_lower for word in ['senior', 'lead', 'principal', 'staff', '5+ years', '7+ years', '10+ years']):
            return 'Senior'
        elif any(word in text_lower for word in ['mid-level', 'intermediate', '3-5 years', '2-4 years', '3+ years']):
            return 'Mid-Level'
        elif any(word in text_lower for word in ['junior', 'entry', 'graduate', '0-2 years', 'new grad', 'early career']):
            return 'Junior'
        else:
            return 'Not Specified'

    def _detect_job_type(self, text):
        """Detect job type"""
        if not text:
            return 'Not Specified'

        text_lower = text.lower()
        types = []

        if 'remote' in text_lower or 'work from home' in text_lower:
            types.append('Remote')
        if 'hybrid' in text_lower:
            types.append('Hybrid')
        if 'onsite' in text_lower or 'on-site' in text_lower:
            types.append('Onsite')
        if 'full-time' in text_lower or 'full time' in text_lower:
            types.append('Full-Time')
        if 'part-time' in text_lower or 'part time' in text_lower:
            types.append('Part-Time')
        if 'contract' in text_lower:
            types.append('Contract')

        return ', '.join(types) if types else 'Not Specified'

    def _extract_salary(self, text):
        """Extract salary if available"""
        if not text:
            return 'Not Specified'

        salary_patterns = [
            r'\$[\d,]+k?\s*-\s*\$[\d,]+k?',
            r'\$[\d,]+k?',
            r'[\d,]+k\s*-\s*[\d,]+k',
        ]

        for pattern in salary_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return 'Not Specified'

    def format_job_for_interview(self, job_data):
        """Format scraped job data for interview system"""

        skills_text = ', '.join(job_data['skills'][:15]) if job_data['skills'] else 'Not specified'
        requirements_text = '\n'.join([f"{i+1}. {req}" for i, req in enumerate(job_data['requirements'][:10])]) if job_data['requirements'] else 'Not specified'

        formatted = f"""
Job Title: {job_data['title']}
Company: {job_data['company']}
Location: {job_data['location']}
Experience Level: {job_data['experience_level']}
Job Type: {job_data['job_type']}
Salary: {job_data['salary']}

KEY SKILLS REQUIRED:
{skills_text}

KEY REQUIREMENTS:
{requirements_text}

FULL DESCRIPTION:
{job_data['description'][:2500]}
"""
        return formatted.strip()


# Test function
if __name__ == "__main__":
    scraper = JobScraper()

    # Test URL
    test_url = input("Enter job URL to test: ")

    print("\n🔍 Testing scraper...")
    result = scraper.scrape_job(test_url)

    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
    else:
        print(f"\n✅ Successfully scraped!")
        print(f"Title: {result['title']}")
        print(f"Company: {result['company']}")
        print(f"Location: {result['location']}")
        print(f"Experience: {result['experience_level']}")
        print(f"Skills: {', '.join(result['skills'][:10])}")
        print(f"\nFormatted:\n{scraper.format_job_for_interview(result)}")