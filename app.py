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

# Import our new code quality analyzer
from code_quality_analyzer import CodeQualityAnalyzer
from security_scanner import SecurityScanner
from dependency_analyzer import DependencyAnalyzer
from architecture_analyzer import ArchitectureAnalyzer

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
        # Initialize all analyzers
        self.quality_analyzer = CodeQualityAnalyzer()
        self.security_scanner = SecurityScanner()
        self.dependency_analyzer = DependencyAnalyzer()
        self.architecture_analyzer = ArchitectureAnalyzer()
    
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
        
        # NEW: Security Analysis
        quality_metrics["security"] = self.security_scanner.scan_repository(detailed_files)
        
        # NEW: Dependency Analysis
        quality_metrics["dependencies"] = self.dependency_analyzer.analyze_dependencies(detailed_files)
        
        # NEW: Architecture Analysis
        quality_metrics["architecture"] = self.architecture_analyzer.analyze_architecture(detailed_files)
        
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
st.markdown("**Comprehensive GitHub repository analysis with quality metrics, security scanning, dependency analysis, and architecture visualization - all powered by free local AI models!**")

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
    
    st.header("🆕 Comprehensive Analysis Features")
    st.markdown("""
    **🏆 Code Quality Analysis:**
    - 🧮 Complexity metrics
    - 🔄 Duplication detection  
    - 📏 Function size analysis
    - 💬 Comment ratio tracking
    - 🏆 Overall quality score
    
    **🔒 Security Analysis:**
    - 🔐 Hardcoded secrets detection
    - ⚠️ Vulnerability scanning
    - 🚨 Risk level assessment
    - 💡 Security recommendations
    
    **📦 Dependency Analysis:**
    - 🔍 Vulnerability scanning
    - 📈 Outdated package detection
    - 🏥 Health score calculation
    - 🏗️ Multi-ecosystem support
    
    **🏗️ Architecture Analysis:**
    - 📊 Dependency visualization
    - 🔄 Circular dependency detection
    - 🎯 Pattern recognition
    - ⚡ Central component analysis
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
    
    with st.spinner(f"🧠 Performing comprehensive analysis (quality, security, dependencies, architecture)... This may take 2-3 minutes"):
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
        
        # Enhanced analysis tabs WITH ALL NEW TABS
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📁 File Structure", 
            "📄 File Explanations",
            "🤖 AI Insights",
            "⚙️ Technologies",
            "🏆 Code Quality",
            "🔒 Security Scan",  # NEW TAB
            "📦 Dependencies",   # NEW TAB
            "🏗️ Architecture"   # NEW TAB
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
        
        # NEW: Security Analysis Tab
        with tab6:
            st.subheader("🔒 Security Analysis")
            
            security_data = quality_metrics.get('security', {})
            if security_data and security_data.get('total_files_scanned', 0) > 0:
                # Security Overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    risk_level = security_data.get('risk_level', 'unknown')
                    risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
                    risk_emoji = risk_colors.get(risk_level, '⚪')
                    st.metric("Risk Level", f"{risk_emoji} {risk_level.title()}")
                
                with col2:
                    st.metric("Secrets Found", security_data.get('secrets_found', 0))
                
                with col3:
                    st.metric("Vulnerabilities", security_data.get('vulnerabilities_found', 0))
                
                with col4:
                    st.metric("Files Scanned", security_data.get('total_files_scanned', 0))
                
                # Risk Summary Chart
                risk_summary = security_data.get('risk_summary', {})
                if any(risk_summary.values()):
                    st.markdown("### 📊 Risk Distribution")
                    risk_df = pd.DataFrame(
                        list(risk_summary.items()),
                        columns=['Risk Level', 'Count']
                    )
                    risk_df = risk_df[risk_df['Count'] > 0]
                    
                    if not risk_df.empty:
                        fig = px.bar(
                            risk_df, 
                            x='Risk Level', 
                            y='Count',
                            color='Risk Level',
                            color_discrete_map={
                                'low': '#2ECC71',
                                'medium': '#F39C12', 
                                'high': '#E67E22',
                                'critical': '#E74C3C'
                            },
                            title="Security Issues by Risk Level"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Findings
                findings = security_data.get('findings', {})
                
                # Secrets Section
                secrets = findings.get('secrets', [])
                if secrets:
                    st.markdown("### 🔐 Hardcoded Secrets Found")
                    st.warning(f"Found {len(secrets)} potential secrets in your code. These should be moved to environment variables or secure vaults.")
                    
                    for secret in secrets[:10]:  # Show top 10
                        with st.expander(f"🚨 {secret['subtype'].replace('_', ' ').title()} in {secret['file']}"):
                            st.markdown(f"**Line {secret['line']}:** `{secret['content']}`")
                            st.markdown(f"**Risk Level:** {secret['risk_level'].title()}")
                            st.markdown(f"**Recommendation:** {secret['recommendation']}")
                
                # Vulnerabilities Section  
                vulnerabilities = findings.get('vulnerabilities', [])
                if vulnerabilities:
                    st.markdown("### ⚠️ Potential Vulnerabilities")
                    
                    for vuln in vulnerabilities[:10]:  # Show top 10
                        with st.expander(f"⚠️ {vuln['subtype'].replace('_', ' ').title()} in {vuln['file']}"):
                            st.markdown(f"**Line {vuln['line']}:** `{vuln['content']}`")
                            st.markdown(f"**Risk Level:** {vuln['risk_level'].title()}")
                            st.markdown(f"**Description:** {vuln['description']}")
                            st.markdown(f"**Recommendation:** {vuln['recommendation']}")
                
                # Recommendations Section
                recommendations = findings.get('recommendations', [])
                if recommendations:
                    st.markdown("### 💡 Security Recommendations")
                    for rec in recommendations:
                        st.markdown(f'''
                        <div class="recommendation-box">
                        {rec}
                        </div>
                        ''', unsafe_allow_html=True)
            
            else:
                st.info("Security analysis not available. This feature analyzes code files for potential security issues.")
        
        # NEW: Dependencies Analysis Tab
        with tab7:
            st.subheader("📦 Dependencies Analysis")
            
            dependency_data = quality_metrics.get('dependencies', {})
            if dependency_data and dependency_data.get('total_dependencies', 0) > 0:
                
                # Dependencies Overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Dependencies", dependency_data.get('total_dependencies', 0))
                
                with col2:
                    health_score = dependency_data.get('health_score', 0)
                    if health_score >= 90:
                        health_color = "🟢"
                    elif health_score >= 70:
                        health_color = "🟡"
                    else:
                        health_color = "🔴"
                    st.metric("Health Score", f"{health_color} {health_score}/100")
                
                with col3:
                    vuln_count = len(dependency_data.get('vulnerabilities', []))
                    st.metric("Vulnerabilities", vuln_count)
                
                with col4:
                    outdated_count = len(dependency_data.get('outdated_packages', []))
                    st.metric("Outdated Packages", outdated_count)
                
                # Ecosystems Overview
                ecosystems = dependency_data.get('ecosystems', {})
                if ecosystems:
                    st.markdown("### 🏗️ Dependency Ecosystems")
                    
                    ecosystem_df = pd.DataFrame([
                        {
                            'Ecosystem': ecosystem.title(),
                            'File': info['file'],
                            'Dependencies': info['dependency_count']
                        }
                        for ecosystem, info in ecosystems.items()
                    ])
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.dataframe(ecosystem_df, use_container_width=True)
                    
                    with col2:
                        if len(ecosystems) > 1:
                            fig = px.pie(
                                ecosystem_df,
                                values='Dependencies',
                                names='Ecosystem',
                                title="Dependencies by Ecosystem"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Issues Summary
                summary = dependency_data.get('summary', {})
                if any(summary.values()):
                    st.markdown("### ⚠️ Issues Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        critical_vulns = summary.get('critical_vulns', 0)
                        st.metric("Critical Vulnerabilities", critical_vulns)
                    
                    with col2:
                        high_vulns = summary.get('high_vulns', 0)
                        st.metric("High Vulnerabilities", high_vulns)
                    
                    with col3:
                        major_updates = summary.get('outdated_major', 0)
                        st.metric("Major Updates Available", major_updates)
                    
                    with col4:
                        minor_updates = summary.get('outdated_minor', 0)
                        st.metric("Minor Updates Available", minor_updates)
                
                # Vulnerabilities Section
                vulnerabilities = dependency_data.get('vulnerabilities', [])
                if vulnerabilities:
                    st.markdown("### 🚨 Security Vulnerabilities")
                    st.error(f"Found {len(vulnerabilities)} security vulnerabilities in dependencies")
                    
                    for vuln in vulnerabilities[:10]:  # Show top 10
                        severity_emoji = {
                            'critical': '🔥',
                            'high': '🚨', 
                            'medium': '⚠️',
                            'low': '💡'
                        }
                        emoji = severity_emoji.get(vuln.get('severity', 'medium'), '⚠️')
                        
                        with st.expander(f"{emoji} {vuln['package']} v{vuln['current_version']} - {vuln['severity'].title()} Severity"):
                            st.markdown(f"**Description:** {vuln['description']}")
                            st.markdown(f"**Vulnerable Version:** {vuln['vulnerable_version']}")
                            st.markdown(f"**Recommendation:** {vuln['recommendation']}")
                
                # Outdated Packages Section
                outdated_packages = dependency_data.get('outdated_packages', [])
                if outdated_packages:
                    st.markdown("### 📈 Outdated Packages")
                    
                    # Group by update type
                    major_updates = [p for p in outdated_packages if p['update_type'] == 'major']
                    minor_updates = [p for p in outdated_packages if p['update_type'] == 'minor']
                    patch_updates = [p for p in outdated_packages if p['update_type'] == 'patch']
                    
                    if major_updates:
                        st.markdown("#### 🔴 Major Updates (Review Breaking Changes)")
                        for pkg in major_updates[:5]:
                            with st.expander(f"📦 {pkg['package']}: {pkg['current_version']} → {pkg['latest_version']}"):
                                st.markdown(f"**Current:** {pkg['current_version']}")
                                st.markdown(f"**Latest:** {pkg['latest_version']}")
                                st.markdown(f"**Recommendation:** {pkg['recommendation']}")
                    
                    if minor_updates:
                        st.markdown("#### 🟡 Minor Updates (Generally Safe)")
                        for pkg in minor_updates[:5]:
                            st.markdown(f"- **{pkg['package']}**: {pkg['current_version']} → {pkg['latest_version']}")
                    
                    if patch_updates:
                        st.markdown("#### 🟢 Patch Updates (Safe to Update)")
                        for pkg in patch_updates[:5]:
                            st.markdown(f"- **{pkg['package']}**: {pkg['current_version']} → {pkg['latest_version']}")
                
                # Recommendations Section
                recommendations = dependency_data.get('recommendations', [])
                if recommendations:
                    st.markdown("### 💡 Recommendations")
                    for rec in recommendations:
                        st.markdown(f'''
                        <div class="recommendation-box">
                        {rec}
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Dependency Files Found
                dependency_files = dependency_data.get('dependency_files_found', [])
                if dependency_files:
                    st.markdown("### 📄 Dependency Files Analyzed")
                    for file in dependency_files:
                        st.markdown(f"- `{file}`")
            
            else:
                st.info("No dependency files found or dependencies could not be analyzed. This feature looks for files like requirements.txt, package.json, pom.xml, etc.")
        
        # NEW: Architecture Analysis Tab
        with tab8:
            st.subheader("🏗️ Architecture Analysis")
            
            architecture_data = quality_metrics.get('architecture', {})
            if architecture_data and architecture_data.get('dependency_metrics', {}).get('total_modules', 0) > 0:
                
                # Architecture Overview
                complexity_metrics = architecture_data.get('complexity_metrics', {})
                dependency_metrics = architecture_data.get('dependency_metrics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Modules", dependency_metrics.get('total_modules', 0))
                
                with col2:
                    st.metric("Dependencies", dependency_metrics.get('total_dependencies', 0))
                
                with col3:
                    complexity_level = complexity_metrics.get('complexity_level', 'simple')
                    st.metric("Complexity", complexity_level.title())
                
                with col4:
                    circular_count = len(architecture_data.get('circular_dependencies', []))
                    st.metric("Circular Deps", circular_count)
                
                # Architectural Patterns
                patterns = architecture_data.get('architectural_patterns', [])
                if patterns:
                    st.markdown("### 🎯 Detected Architectural Patterns")
                    for pattern in patterns:
                        confidence_color = "🟢" if pattern['confidence'] > 0.7 else "🟡" if pattern['confidence'] > 0.5 else "🔴"
                        st.markdown(f"**{confidence_color} {pattern['pattern']}** (Confidence: {pattern['confidence']:.0%})")
                        st.markdown(f"   {pattern['description']}")
                        
                        # Show evidence
                        evidence = pattern.get('evidence', {})
                        if evidence:
                            evidence_str = ", ".join([f"{k}: {v}" for k, v in evidence.items()])
                            st.markdown(f"   *Evidence: {evidence_str}*")
                else:
                    st.info("No specific architectural patterns detected. Consider adopting patterns like MVC or layered architecture for better organization.")
                
                # Central Components Analysis
                central_components = architecture_data.get('central_components', [])
                if central_components:
                    st.markdown("### ⚡ Most Important Components")
                    st.markdown("These components are central to your architecture - they connect many parts together.")
                    
                    # Create a DataFrame for better display
                    central_df = pd.DataFrame([
                        {
                            'Module': comp['module'],
                            'Connections': comp['degree_centrality'],
                            'Fan-In': comp['fan_in'],
                            'Fan-Out': comp['fan_out'],
                            'Type': comp['centrality_type'].title()
                        }
                        for comp in central_components[:10]
                    ])
                    
                    st.dataframe(central_df, use_container_width=True)
                    
                    # Visualization of centrality
                    if len(central_components) > 1:
                        fig = px.scatter(
                            central_df,
                            x='Fan-Out',
                            y='Fan-In', 
                            size='Connections',
                            color='Type',
                            hover_data=['Module'],
                            title="Component Centrality Analysis",
                            labels={'Fan-In': 'Modules that depend on this', 'Fan-Out': 'Modules this depends on'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Dependency Metrics Deep Dive
                st.markdown("### 📊 Dependency Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Coupling Analysis:**")
                    avg_fan_out = dependency_metrics.get('average_fan_out', 0)
                    avg_fan_in = dependency_metrics.get('average_fan_in', 0)
                    
                    st.metric("Average Fan-Out", f"{avg_fan_out:.1f}")
                    st.metric("Average Fan-In", f"{avg_fan_in:.1f}")
                    
                    # Coupling assessment
                    if avg_fan_out > 5:
                        st.warning("⚠️ High fan-out detected - consider reducing dependencies")
                    elif avg_fan_out < 2:
                        st.success("✅ Good dependency management")
                    else:
                        st.info("📊 Moderate coupling levels")
                
                with col2:
                    st.markdown("**Structural Metrics:**")
                    density = complexity_metrics.get('graph_density', 0)
                    modularity = complexity_metrics.get('modularity', 0)
                    
                    st.metric("Graph Density", f"{density:.3f}")
                    st.metric("Modularity", f"{modularity:.3f}")
                    
                    # Structural assessment
                    if density > 0.3:
                        st.warning("⚠️ Very dense architecture - consider simplification")
                    elif modularity > 0.5:
                        st.success("✅ Good modular structure")
                
                # Circular Dependencies
                circular_deps = architecture_data.get('circular_dependencies', [])
                if circular_deps:
                    st.markdown("### 🔄 Circular Dependencies")
                    st.error(f"Found {len(circular_deps)} circular dependencies that should be resolved:")
                    
                    for i, cycle in enumerate(circular_deps[:5], 1):  # Show first 5
                        cycle_str = " → ".join(cycle)
                        st.markdown(f"**{i}.** `{cycle_str}`")
                        
                        with st.expander(f"How to fix cycle {i}"):
                            st.markdown("""
                            **Strategies to break circular dependencies:**
                            1. **Dependency Inversion**: Create an interface that both modules can depend on
                            2. **Extract Common Logic**: Move shared code to a separate module
                            3. **Event-Driven Architecture**: Use events instead of direct dependencies
                            4. **Facade Pattern**: Create a facade that manages the interaction
                            """)
                
                # Most Coupled Modules
                most_coupled = dependency_metrics.get('most_coupled_modules', [])
                if most_coupled:
                    st.markdown("### 🔗 Most Coupled Modules")
                    st.markdown("These modules have the highest number of connections:")
                    
                    for module, metrics in most_coupled[:5]:
                        total_coupling = metrics['total_coupling']
                        instability = metrics['instability']
                        
                        # Determine stability status
                        if instability < 0.3:
                            stability_status = "🟢 Stable"
                        elif instability < 0.7:
                            stability_status = "🟡 Moderate"
                        else:
                            stability_status = "🔴 Unstable"
                        
                        st.markdown(f"**{module}** - {total_coupling} connections ({stability_status})")
                
                # Recommendations
                recommendations = architecture_data.get('recommendations', [])
                if recommendations:
                    st.markdown("### 💡 Architecture Recommendations")
                    for rec in recommendations:
                        st.markdown(f'''
                        <div class="recommendation-box">
                        {rec}
                        </div>
                        ''', unsafe_allow_html=True)
            
            else:
                st.info("Architecture analysis requires code files with import statements. This feature works best with Python, JavaScript, TypeScript, and similar languages.")
    else:
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")