# Enhanced app.py - Repository analyzer with code quality metrics

import streamlit as st
import asyncio
import json
import time
import os
import tempfile
import shutil
import re
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import git

# Import our code quality analyzer
from code_quality_analyzer import CodeQualityAnalyzer
from visual_code_analyzer import VisualCodeAnalyzer

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Free LLM Repository Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with new styles for quality metrics
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #27AE60;
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-error {
        color: #E74C3C;
        background-color: #FADBD8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
    }
    .cost-free {
        color: #27AE60;
        font-weight: bold;
        font-size: 1.2em;
    }
    .quality-score-a {
        background: linear-gradient(135deg, #27AE60, #2ECC71);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    .quality-score-b {
        background: linear-gradient(135deg, #3498DB, #5DADE2);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    .quality-score-c {
        background: linear-gradient(135deg, #F39C12, #F4D03F);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    .quality-score-d {
        background: linear-gradient(135deg, #E67E22, #F8C471);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    .quality-score-f {
        background: linear-gradient(135deg, #E74C3C, #EC7063);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    .recommendation-box {
        background-color: #EBF3FD;
        border-left: 4px solid #3498DB;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

class RepositoryAnalyzer:
    """Enhanced repository analyzer with code quality metrics"""
    
    def __init__(self):
        self.ollama_available = self._check_ollama()
        # Initialize analyzers
        self.quality_analyzer = CodeQualityAnalyzer()
        self.visual_analyzer = VisualCodeAnalyzer()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    async def _call_ollama(self, messages: List[Dict], model: str = "llama3.2:3b") -> str:
        """Call Ollama API"""
        import subprocess
        import json
        
        # Create prompt from messages
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
        
        prompt += "Assistant:"
        
        try:
            # Call ollama
            cmd = ['ollama', 'run', model]
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            stdout, stderr = process.communicate(input=prompt, timeout=120)
            
            if process.returncode == 0:
                return stdout.strip()
            else:
                return f"Error: {stderr}"
                
        except subprocess.TimeoutExpired:
            process.kill()
            return "Error: Request timed out"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _parse_repo_url(self, repo_input: str) -> Optional[Dict[str, str]]:
        """Parse repository URL in multiple formats"""
        repo_input = repo_input.strip()
        
        patterns = [
            r'github\.com/([^/\s]+)/([^/\s]+)',  # GitHub URL
            r'^([a-zA-Z0-9\-_]+)/([a-zA-Z0-9\-_.]+)$'  # Direct owner/repo
        ]
        
        for pattern in patterns:
            match = re.search(pattern, repo_input)
            if match:
                return {
                    "owner": match.group(1),
                    "repo": match.group(2).rstrip('.git')
                }
        return None
    
    def _clone_repository(self, repo_info: Dict[str, str]) -> str:
        """Clone repository to analyze"""
        temp_dir = tempfile.mkdtemp()
        repo_url = f"https://github.com/{repo_info['owner']}/{repo_info['repo']}.git"
        
        try:
            git.Repo.clone_from(repo_url, temp_dir, depth=1)
            return temp_dir
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise Exception(f"Failed to clone repository: {e}")
    
    def _extract_imports(self, content: str, file_ext: str) -> List[str]:
        """Extract import statements from code"""
        imports = []
        lines = content.split('\n')
        
        for line in lines[:50]:
            line = line.strip()
            if file_ext == '.py':
                if line.startswith('import ') or line.startswith('from '):
                    imports.append(line)
            elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
                if line.startswith('import ') or ('require(' in line and 'const ' in line):
                    imports.append(line)
        
        return imports[:10]
    
    def _extract_functions(self, content: str, file_ext: str) -> List[str]:
        """Extract function definitions from code"""
        functions = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if file_ext == '.py':
                if line.startswith('def ') and '(' in line:
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    functions.append(func_name)
            elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
                if ('function ' in line and '(' in line) or ('=>' in line and '(' in line):
                    if 'function ' in line:
                        func_name = line.split('function ')[1].split('(')[0].strip()
                    else:
                        func_name = line.split('=')[0].strip() if '=' in line else 'anonymous'
                    functions.append(func_name)
        
        return functions[:15]
    
    def _extract_classes(self, content: str, file_ext: str) -> List[str]:
        """Extract class definitions from code"""
        classes = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if file_ext == '.py':
                if line.startswith('class ') and ':' in line:
                    class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                    classes.append(class_name)
            elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
                if line.startswith('class ') and '{' in line:
                    class_name = line.split('class ')[1].split('{')[0].split(' extends')[0].strip()
                    classes.append(class_name)
        
        return classes[:10]
    
    def _analyze_code_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure and collect file details with quality metrics"""
        analysis = {
            "file_structure": {},
            "technologies": [],
            "key_files": [],
            "detailed_files": {},
            "quality_metrics": {}  # NEW: Add quality metrics to analysis
        }
        
        file_types = {}
        total_files = 0
        total_lines = 0
        detailed_files = {}
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common irrelevant folders
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                relative_path = os.path.relpath(file_path, repo_path)
                
                file_types[file_ext] = file_types.get(file_ext, 0) + 1
                total_files += 1
                
                # Analyze code files
                if file_ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php', '.cs', '.jsx', '.tsx']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = len(content.splitlines())
                            total_lines += lines
                            
                            if lines > 0:
                                detailed_files[relative_path] = {
                                    "extension": file_ext,
                                    "lines": lines,
                                    "content_preview": content[:2000],
                                    "full_content": content,  # Store full content for quality analysis
                                    "imports": self._extract_imports(content, file_ext),
                                    "functions": self._extract_functions(content, file_ext),
                                    "classes": self._extract_classes(content, file_ext)
                                }
                    except Exception:
                        continue
                
                # Analyze important non-code files
                elif file_ext in ['.md', '.txt', '.json', '.yaml', '.yml'] or file in ['README', 'LICENSE']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if len(content.strip()) > 0:
                                detailed_files[relative_path] = {
                                    "extension": file_ext,
                                    "lines": len(content.splitlines()),
                                    "content_preview": content[:1000],
                                    "file_type": "configuration" if file_ext in ['.json', '.yaml', '.yml'] else "documentation"
                                }
                    except Exception:
                        continue
                
                # Identify key files
                if file.lower() in ['readme.md', 'requirements.txt', 'package.json', 'dockerfile', 'setup.py']:
                    analysis["key_files"].append(relative_path)
        
        analysis["file_structure"] = {
            "total_files": total_files,
            "file_types": file_types,
            "total_lines": total_lines
        }
        
        # Simple technology detection
        technologies = []
        if '.py' in file_types: technologies.append("Python")
        if '.js' in file_types or '.jsx' in file_types: technologies.append("JavaScript")
        if '.ts' in file_types or '.tsx' in file_types: technologies.append("TypeScript")
        if '.java' in file_types: technologies.append("Java")
        if 'package.json' in [f.lower() for f in analysis["key_files"]]: technologies.append("Node.js")
        if 'requirements.txt' in [f.lower() for f in analysis["key_files"]]: technologies.append("Python Package")
        
        analysis["technologies"] = technologies
        analysis["detailed_files"] = detailed_files
        
        # NEW: Analyze code quality metrics
        analysis["quality_metrics"] = self._analyze_quality_metrics(detailed_files)
        
        return analysis
    
    def _analyze_quality_metrics(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze comprehensive code quality metrics for the entire repository.
        This is where we put on our code detective hat and look for all types of quality clues!
        """
        quality_metrics = {
            "complexity": {"total_files_analyzed": 0, "average_complexity": 0, "functions": []},
            "duplication": {},
            "function_sizes": {},
            "comment_ratio": 0,
            "security": {},  # NEW: Security analysis
            "dependencies": {},  # NEW: Dependency analysis
            "architecture": {},  # NEW: Architecture analysis
            "overall_score": {}
        }
        
        # Analyze complexity for Python files
        total_complexity = 0
        complexity_files = 0
        all_functions = []
        total_comment_ratio = 0
        comment_files = 0
        
        for file_path, file_info in detailed_files.items():
            file_ext = file_info.get('extension', '')
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            
            if not content:
                continue
            
            # Calculate complexity (currently only for Python)
            if file_ext == '.py':
                complexity_result = self.quality_analyzer.calculate_cyclomatic_complexity(content, file_ext)
                if not complexity_result.get('error'):
                    complexity_files += 1
                    file_complexity = complexity_result.get('average_complexity', 0)
                    total_complexity += file_complexity
                    all_functions.extend(complexity_result.get('functions', []))
            
            # Calculate comment ratio for all code files
            if file_ext in ['.py', '.js', '.ts', '.java', '.jsx', '.tsx']:
                comment_ratio = self.quality_analyzer.calculate_comment_ratio(content, file_ext)
                total_comment_ratio += comment_ratio
                comment_files += 1
        
        # Calculate averages
        if complexity_files > 0:
            quality_metrics["complexity"]["average_complexity"] = total_complexity / complexity_files
            quality_metrics["complexity"]["total_files_analyzed"] = complexity_files
            quality_metrics["complexity"]["functions"] = all_functions
        
        if comment_files > 0:
            quality_metrics["comment_ratio"] = total_comment_ratio / comment_files
        
        # Analyze code duplication
        quality_metrics["duplication"] = self.quality_analyzer.detect_code_duplication(detailed_files)
        
        # Analyze function sizes
        quality_metrics["function_sizes"] = self.quality_analyzer.analyze_function_sizes(detailed_files)
        
        # NEW: Visual Analysis
        quality_metrics["visual_analysis"] = self.visual_analyzer.analyze_code_visually(detailed_files)
        
        # Generate overall quality score (now includes all metrics)
        quality_metrics["overall_score"] = self.quality_analyzer.generate_quality_score(quality_metrics)
        
        return quality_metrics
    
    async def _explain_single_file(self, file_path: str, file_info: Dict[str, Any]) -> str:
        """Generate explanation for a single file"""
        if not self.ollama_available:
            return f"File analysis requires Ollama to be running. File has {file_info.get('lines', 0)} lines."
        
        context = f"""
File: {file_path}
Extension: {file_info.get('extension', 'unknown')}
Lines: {file_info.get('lines', 0)}

Content Preview:
{file_info.get('content_preview', '')[:1500]}

Imports: {', '.join(file_info.get('imports', [])[:5])}
Functions: {', '.join(file_info.get('functions', [])[:5])}
Classes: {', '.join(file_info.get('classes', [])[:5])}
"""
        
        prompt = f"""
Explain the purpose and functionality of this file clearly.

{context}

Provide a concise explanation (2-3 sentences) of:
1. What this file does
2. Its role in the project
3. Key functionality (if code file)

Keep it brief and focused.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a code analyst explaining file purposes concisely."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._call_ollama(messages)
            return response.strip()
            
        except Exception as e:
            return f"File analysis failed: {str(e)}"
    
    async def _generate_file_explanations(self, detailed_files: Dict[str, Any]) -> Dict[str, str]:
        """Generate explanations for individual files using LLM"""
        if not self.ollama_available:
            return {}
        
        file_explanations = {}
        
        # Process important files first (limit to 20 files)
        important_files = []
        regular_files = []
        
        for file_path, file_info in detailed_files.items():
            if (file_path.lower().endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.java')) and 
                file_info.get('lines', 0) > 10) or file_path.lower() in ['readme.md', 'setup.py', 'package.json']:
                important_files.append((file_path, file_info))
            else:
                regular_files.append((file_path, file_info))
        
        files_to_process = important_files[:15] + regular_files[:5]
        
        for file_path, file_info in files_to_process:
            try:
                explanation = await self._explain_single_file(file_path, file_info)
                if explanation:
                    file_explanations[file_path] = explanation
            except Exception as e:
                file_explanations[file_path] = f"Code file ({file_info.get('lines', 0)} lines) - Analysis failed"
        
        return file_explanations
    
    async def _generate_overall_insights(self, analysis: Dict[str, Any]) -> str:
        """Generate overall insights about the repository including quality assessment"""
        if not self.ollama_available:
            return "Overall analysis requires Ollama to be running."
        
        file_structure = analysis['file_structure']
        quality_metrics = analysis.get('quality_metrics', {})
        
        # Build context including quality metrics
        context = f"""
Repository Analysis Results:

FILE STRUCTURE:
- Total files: {file_structure['total_files']}
- Total lines of code: {file_structure['total_lines']}
- File types: {', '.join([f"{ext}({count})" for ext, count in file_structure['file_types'].items()])}

TECHNOLOGIES:
{', '.join(analysis['technologies']) if analysis['technologies'] else 'None detected'}

KEY FILES:
{', '.join(analysis['key_files']) if analysis['key_files'] else 'None detected'}

CODE QUALITY METRICS:
- Average complexity: {quality_metrics.get('complexity', {}).get('average_complexity', 'N/A')}
- Code duplication: {quality_metrics.get('duplication', {}).get('duplication_percentage', 'N/A')}%
- Comment ratio: {quality_metrics.get('comment_ratio', 'N/A')}%
- Quality score: {quality_metrics.get('overall_score', {}).get('score', 'N/A')}/100 (Grade: {quality_metrics.get('overall_score', {}).get('grade', 'N/A')})
"""
        
        prompt = f"""
Based on this repository analysis including code quality metrics, provide a comprehensive assessment.

{context}

Please provide:

1. **Overall Assessment** - What type of project is this and what's its scope?
2. **Code Quality Observations** - What can you infer about code organization and quality?
3. **Key Recommendations** - What improvements could be made?

Keep it concise but insightful, and include observations about the code quality metrics.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a senior software architect analyzing a codebase including quality metrics."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._call_ollama(messages)
            return response.strip()
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    async def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """Analyze a GitHub repository with enhanced quality metrics"""
        # Parse repository info
        repo_info = self._parse_repo_url(repo_url)
        if not repo_info:
            return {"error": "Invalid repository URL format", "success": False}
        
        try:
            # Clone and analyze repository
            repo_path = self._clone_repository(repo_info)
            analysis = await asyncio.get_event_loop().run_in_executor(
                None, self._analyze_code_structure, repo_path
            )
            
            # Generate file explanations
            file_explanations = await self._generate_file_explanations(analysis["detailed_files"])
            analysis["file_explanations"] = file_explanations
            
            # Generate overall insights (now includes quality metrics)
            insights = await self._generate_overall_insights(analysis)
            
            return {
                "repository": f"{repo_info['owner']}/{repo_info['repo']}",
                "analysis": analysis,
                "insights": insights,
                "success": True
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
        
        finally:
            if 'repo_path' in locals():
                shutil.rmtree(repo_path, ignore_errors=True)

# Main Streamlit App (Enhanced with Quality Metrics UI)
st.markdown('<h1 class="main-title">🚀 Free LLM Repository Analyzer</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive GitHub repository analysis with quality metrics and interactive visual diagrams - all powered by free local AI models!**")

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return RepositoryAnalyzer()

analyzer = get_analyzer()

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Ollama status
    if analyzer.ollama_available:
        st.markdown('<div class="status-success">✅ Ollama is running</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-error">❌ Ollama not available</div>', unsafe_allow_html=True)
        st.markdown("""
        **To enable file explanations:**
        1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
        2. Start service: `ollama serve`
        3. Pull model: `ollama pull llama3.2:3b`
        """)
    
    st.header("🎯 Analysis Features")
    st.markdown("""
    **🏆 Code Quality Analysis:**
    - 🧮 Complexity metrics
    - 🔄 Duplication detection  
    - 📏 Function size analysis
    - 💬 Comment ratio tracking
    - 🏆 Overall quality score
    
    **🎨 Visual Code Analysis:**
    - 📊 System flow diagrams
    - 🔄 Component interactions
    - 📈 Data flow visualization
    - 🏗️ Architecture overview
    """)

# Main interface (unchanged)
repo_url = st.text_input(
    "🔗 GitHub Repository URL",
    placeholder="https://github.com/karpathy/micrograd or karpathy/micrograd",
    help="Enter a GitHub repository URL or owner/repo format"
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_button = st.button("🧠 Analyze Repository", type="primary")
with col2:
    clear_button = st.button("🗑️ Clear Results")

# Clear results if requested
if clear_button:
    if 'analysis_results' in st.session_state:
        del st.session_state['analysis_results']
    st.rerun()

# Perform analysis
if analyze_button and repo_url:
    if not analyzer.ollama_available:
        st.warning("⚠️ Ollama is not running. Analysis will work but file explanations will be limited.")
    
    with st.spinner(f"🧠 Performing comprehensive analysis with visual diagrams... This may take 2-3 minutes"):
        start_time = time.time()
        
        try:
            result = asyncio.run(analyzer.analyze_repository(repo_url))
            end_time = time.time()
            analysis_time = end_time - start_time
            
            st.session_state['analysis_results'] = result
            st.session_state['analysis_time'] = analysis_time
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Display results (ENHANCED WITH QUALITY METRICS)
if 'analysis_results' in st.session_state:
    result = st.session_state['analysis_results']
    
    if result.get('success', False):
        analysis_time = st.session_state.get('analysis_time', 0)
        
        st.markdown(f'''
        <div class="status-success">
        ✅ Analysis completed in {analysis_time:.1f} seconds<br/>
        <span class="cost-free">💰 Cost: $0.00 (FREE!)</span>
        </div>
        ''', unsafe_allow_html=True)
        
        # Repository info
        st.header(f"📊 Analysis Results: {result['repository']}")
        
        analysis = result['analysis']
        quality_metrics = analysis.get('quality_metrics', {})
        
        # Enhanced metrics overview WITH QUALITY SCORE
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-box">
            <h3>{analysis["file_structure"]["total_files"]}</h3>
            <p>Total Files</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-box">
            <h3>{analysis["file_structure"]["total_lines"]:,}</h3>
            <p>Lines of Code</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-box">
            <h3>{len(analysis["technologies"])}</h3>
            <p>Technologies</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # NEW: Quality Score Display
        with col4:
            overall_score = quality_metrics.get('overall_score', {})
            score = overall_score.get('score', 0)
            grade = overall_score.get('grade', 'N/A')
            grade_class = f"quality-score-{grade.lower()}" if grade != 'N/A' else "metric-box"
            
            st.markdown(f'''
            <div class="{grade_class}">
            <h3>{score}/100</h3>
            <p>Quality Score ({grade})</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📁 File Structure", 
            "📄 File Explanations",
            "🤖 AI Insights",
            "⚙️ Technologies",
            "🏆 Code Quality",
            "🎨 Visual Analysis"
        ])
        
        # Existing tabs remain the same...
        with tab1:
            st.subheader("File Type Distribution")
            
            if analysis['file_structure']['file_types']:
                file_types_df = pd.DataFrame(
                    list(analysis['file_structure']['file_types'].items()),
                    columns=['Extension', 'Count']
                )
                file_types_df = file_types_df.sort_values('Count', ascending=False)
                
                fig = px.pie(
                    file_types_df, 
                    values='Count', 
                    names='Extension', 
                    title="File Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(file_types_df, use_container_width=True)
        
        with tab2:
            st.subheader("📄 File-by-File Explanations")
            
            file_explanations = analysis.get('file_explanations', {})
            
            if file_explanations:
                # Search functionality
                search_term = st.text_input("🔍 Search files:", placeholder="Type to filter files...")
                
                # Filter files
                filtered_files = {}
                if search_term:
                    for file_path, explanation in file_explanations.items():
                        if search_term.lower() in file_path.lower() or search_term.lower() in explanation.lower():
                            filtered_files[file_path] = explanation
                else:
                    filtered_files = file_explanations
                
                if filtered_files:
                    # Group files by type
                    code_files = {}
                    config_files = {}
                    doc_files = {}
                    
                    for file_path, explanation in filtered_files.items():
                        if file_path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.go', '.rs')):
                            code_files[file_path] = explanation
                        elif file_path.endswith(('.json', '.yaml', '.yml', '.toml', '.cfg')):
                            config_files[file_path] = explanation
                        else:
                            doc_files[file_path] = explanation
                    
                    # Display each category
                    if code_files:
                        st.markdown("### 💻 Code Files")
                        for file_path, explanation in code_files.items():
                            with st.expander(f"📄 {file_path}"):
                                st.markdown(explanation)
                    
                    if config_files:
                        st.markdown("### ⚙️ Configuration Files")
                        for file_path, explanation in config_files.items():
                            with st.expander(f"🔧 {file_path}"):
                                st.markdown(explanation)
                    
                    if doc_files:
                        st.markdown("### 📚 Documentation Files")
                        for file_path, explanation in doc_files.items():
                            with st.expander(f"📖 {file_path}"):
                                st.markdown(explanation)
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Code Files", len(code_files))
                with col2:
                    st.metric("⚙️ Config Files", len(config_files))
                with col3:
                    st.metric("📚 Doc Files", len(doc_files))
            else:
                st.info("File explanations not available. Make sure Ollama is running and re-run the analysis.")
        
        with tab3:
            st.subheader("AI-Generated Insights")
            
            insights = result.get('insights', '')
            if insights:
                st.markdown(insights)
            else:
                st.info("AI insights not available. Make sure Ollama is running.")
        
        with tab4:
            st.subheader("Technology Stack")
            
            if analysis['technologies']:
                for tech in analysis['technologies']:
                    st.markdown(f"🔧 **{tech}**")
            else:
                st.info("No specific technologies detected")
            
            if analysis.get('key_files'):
                st.subheader("📋 Key Files Found")
                for file in analysis['key_files']:
                    st.markdown(f"📄 `{file}`")
        
        # NEW: Code Quality Tab
        with tab5:
            st.subheader("🏆 Code Quality Analysis")
            
            if quality_metrics:
                # Overall Score Section
                overall_score = quality_metrics.get('overall_score', {})
                if overall_score:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        score = overall_score.get('score', 0)
                        grade = overall_score.get('grade', 'N/A')
                        grade_class = f"quality-score-{grade.lower()}" if grade != 'N/A' else "metric-box"
                        
                        st.markdown(f'''
                        <div class="{grade_class}">
                        <h2>Overall Quality Score</h2>
                        <h1>{score}/100 ({grade})</h1>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**🎯 Recommendations:**")
                        recommendations = overall_score.get('recommendations', [])
                        for rec in recommendations:
                            st.markdown(f'''
                            <div class="recommendation-box">
                            {rec}
                            </div>
                            ''', unsafe_allow_html=True)
                
                # Detailed Metrics
                st.markdown("---")
                
                # Complexity Analysis
                complexity_data = quality_metrics.get('complexity', {})
                if complexity_data.get('functions'):
                    st.markdown("### 🧮 Complexity Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Complexity", f"{complexity_data.get('average_complexity', 0):.1f}")
                        st.metric("Files Analyzed", complexity_data.get('total_files_analyzed', 0))
                    
                    with col2:
                        # Complexity distribution chart
                        functions = complexity_data.get('functions', [])
                        if functions:
                            complexity_df = pd.DataFrame(functions)
                            fig = px.histogram(
                                complexity_df, 
                                x='complexity', 
                                title="Function Complexity Distribution",
                                nbins=10
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # High complexity functions
                    high_complexity = [f for f in functions if f.get('complexity', 0) > 10]
                    if high_complexity:
                        st.markdown("**🚨 High Complexity Functions (>10):**")
                        for func in high_complexity[:10]:  # Show top 10
                            st.markdown(f"- `{func['name']}` (complexity: {func['complexity']}, line: {func['line']})")
                
                # Duplication Analysis
                duplication_data = quality_metrics.get('duplication', {})
                if duplication_data:
                    st.markdown("### ♻️ Code Duplication Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duplicate Blocks", duplication_data.get('total_duplicates', 0))
                    with col2:
                        st.metric("Duplication %", f"{duplication_data.get('duplication_percentage', 0):.1f}%")
                    with col3:
                        duplication_pct = duplication_data.get('duplication_percentage', 0)
                        if duplication_pct < 5:
                            status = "✅ Low"
                        elif duplication_pct < 15:
                            status = "⚠️ Medium"
                        else:
                            status = "🚨 High"
                        st.metric("Status", status)
                
                # Function Size Analysis
                function_sizes = quality_metrics.get('function_sizes', {})
                if function_sizes.get('total_functions', 0) > 0:
                    st.markdown("### 📏 Function Size Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Function Size", f"{function_sizes.get('average_size', 0):.1f} lines")
                        st.metric("Total Functions", function_sizes.get('total_functions', 0))
                    
                    with col2:
                        # Function size distribution
                        size_dist = function_sizes.get('size_distribution', {})
                        if size_dist:
                            labels = list(size_dist.keys())
                            values = list(size_dist.values())
                            
                            fig = px.pie(
                                values=values, 
                                names=labels, 
                                title="Function Size Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Comment Ratio
                comment_ratio = quality_metrics.get('comment_ratio', 0)
                if comment_ratio > 0:
                    st.markdown("### 💬 Comment Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Comment Ratio", f"{comment_ratio:.1f}%")
                    with col2:
                        if comment_ratio < 10:
                            status = "🚨 Low - Add more comments"
                        elif comment_ratio < 30:
                            status = "✅ Good"
                        else:
                            status = "⚠️ High - Consider reducing"
                        st.metric("Status", status)
            
            else:
                st.info("Code quality analysis not available. This feature works best with Python repositories.")
        
        # NEW: Visual Analysis Tab
        with tab6:
            st.subheader("🎨 Visual Code Analysis")
            
            visual_data = quality_metrics.get('visual_analysis', {})
            if visual_data and visual_data.get('entry_points'):
                
                # System Overview
                system_overview = visual_data.get('system_overview', {})
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Entry Points", system_overview.get('entry_points', 0))
                
                with col2:
                    st.metric("Main Flows", system_overview.get('main_flows', 0))
                
                with col3:
                    st.metric("Component Interactions", system_overview.get('component_interactions', 0))
                
                with col4:
                    complexity = system_overview.get('complexity_level', 'Unknown')
                    st.metric("Complexity", complexity)
                
                # Entry Points Section
                entry_points = visual_data.get('entry_points', [])
                if entry_points:
                    st.markdown("### 🎯 Application Entry Points")
                    st.markdown("These are the main ways to start or access your application:")
                    
                    for ep in entry_points:
                        entry_type_emoji = {
                            'main_script': '🚀',
                            'main_function': '🔧',
                            'api_endpoint': '🌐',
                            'streamlit_app': '📊',
                            'react_component': '⚛️'
                        }
                        emoji = entry_type_emoji.get(ep['type'], '📁')
                        
                        with st.expander(f"{emoji} {ep['name']} ({ep['type'].replace('_', ' ').title()})"):
                            st.markdown(f"**File:** `{ep['file']}`")
                            st.markdown(f"**Description:** {ep['description']}")
                            if ep.get('line', 1) > 1:
                                st.markdown(f"**Line:** {ep['line']}")
                
                # Visual Diagrams Section
                st.markdown("### 📊 Interactive Diagrams")
                
                visual_diagrams = visual_data.get('visual_diagrams', {})
                
                # System Flow Diagram
                system_flow = visual_diagrams.get('system_flow')
                if system_flow and system_flow != "graph TD\n    A[\"No main flows detected\"]":
                    st.markdown("#### 🌊 System Flow Diagram")
                    st.markdown("This shows how your application flows from start to finish:")
                    
                    try:
                        from streamlit_mermaid import st_mermaid
                        st_mermaid(system_flow, height=400)
                    except ImportError:
                        # Fallback to displaying the mermaid code
                        st.code(system_flow, language='text')
                        st.info("💡 Install streamlit-mermaid for interactive diagrams: `pip install streamlit-mermaid`")
                
                # Component Interaction Diagram
                component_interaction = visual_diagrams.get('component_interaction')
                if component_interaction and component_interaction != "graph TD\n    A[\"No component interactions detected\"]":
                    st.markdown("#### 🔗 Component Interaction Diagram")
                    st.markdown("This shows how different parts of your code work together:")
                    
                    try:
                        from streamlit_mermaid import st_mermaid
                        st_mermaid(component_interaction, height=400)
                    except ImportError:
                        st.code(component_interaction, language='text')
                        st.info("💡 Install streamlit-mermaid for interactive diagrams: `pip install streamlit-mermaid`")
                
                # Data Flow Diagram
                data_flow = visual_diagrams.get('data_flow')
                if data_flow and data_flow != "graph TD\n    A[\"No data flows detected\"]":
                    st.markdown("#### 📈 Data Flow Diagram")
                    st.markdown("This shows how data moves through your application:")
                    
                    try:
                        from streamlit_mermaid import st_mermaid
                        st_mermaid(data_flow, height=400)
                    except ImportError:
                        st.code(data_flow, language='text')
                        st.info("💡 Install streamlit-mermaid for interactive diagrams: `pip install streamlit-mermaid`")
                
                # Main Flows Analysis
                main_flows = visual_data.get('main_flows', [])
                if main_flows:
                    st.markdown("### 🌊 Execution Flow Analysis")
                    
                    for i, flow in enumerate(main_flows[:3], 1):  # Show top 3 flows
                        entry_point = flow['entry_point']
                        flow_steps = flow['flow_steps']
                        
                        with st.expander(f"Flow {i}: {entry_point['name']} ({len(flow_steps)} steps)"):
                            st.markdown(f"**Starting from:** `{entry_point['file']}`")
                            st.markdown(f"**Components involved:** {', '.join(flow['components_involved'][:5])}")
                            
                            # Show key steps
                            st.markdown("**Key execution steps:**")
                            for step in flow_steps[:8]:  # Show first 8 steps
                                step_emoji = {'entry': '🚀', 'import': '📦', 'function_call': '⚡', 'conditional': '🤔'}.get(step['type'], '▶️')
                                st.markdown(f"- {step_emoji} {step['action']}")
                
                # Component Interactions Analysis
                component_interactions = visual_data.get('component_interactions', [])
                if component_interactions:
                    st.markdown("### 🔗 Component Interactions")
                    st.markdown(f"Found {len(component_interactions)} interactions between components:")
                    
                    # Group interactions by component
                    interaction_groups = {}
                    for interaction in component_interactions:
                        from_comp = interaction['from_component']
                        if from_comp not in interaction_groups:
                            interaction_groups[from_comp] = []
                        interaction_groups[from_comp].append(interaction['to_component'])
                    
                    # Display top interacting components
                    for comp, targets in sorted(interaction_groups.items(), key=lambda x: len(x[1]), reverse=True)[:8]:
                        st.markdown(f"**{comp}** → {', '.join(targets[:5])}")
                        if len(targets) > 5:
                            st.markdown(f"   ... and {len(targets)-5} more")
                
                # Data Flow Analysis
                data_flows = visual_data.get('data_flows', [])
                if data_flows:
                    st.markdown("### 📈 Data Flow Analysis")
                    
                    for flow in data_flows[:5]:  # Show top 5 data flows
                        file_name = Path(flow['file']).name
                        inputs = flow['inputs']
                        transformations = flow['transformations']
                        outputs = flow['outputs']
                        
                        if inputs or transformations or outputs:
                            with st.expander(f"📄 {file_name}"):
                                if inputs:
                                    st.markdown(f"**📥 Inputs:** {', '.join(inputs)}")
                                if transformations:
                                    st.markdown(f"**⚙️ Transformations:** {', '.join(transformations)}")
                                if outputs:
                                    st.markdown(f"**📤 Outputs:** {', '.join(outputs)}")
                
                # Insights Section
                insights = visual_data.get('insights', [])
                if insights:
                    st.markdown("### 💡 Visual Structure Insights")
                    for insight in insights:
                        st.markdown(f"- {insight}")
                
                # Recommendations Section
                recommendations = visual_data.get('recommendations', [])
                if recommendations:
                    st.markdown("### 🎯 Recommendations")
                    for rec in recommendations:
                        st.markdown(f'''
                        <div class="recommendation-box">
                        {rec}
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Architecture Style
                arch_style = system_overview.get('architecture_style', 'Unknown')
                if arch_style != 'Unknown':
                    st.markdown("### 🏗️ Architecture Style")
                    st.info(f"**Detected Architecture:** {arch_style}")
            
            else:
                st.info("📊 Visual analysis will show interactive diagrams of your code structure. Analyzing...")
    else:
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")