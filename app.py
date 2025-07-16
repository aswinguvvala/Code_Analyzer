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

# Check for streamlit-mermaid availability
try:
    import streamlit_mermaid as st_mermaid
    MERMAID_AVAILABLE = True
    # Test if st_mermaid actually works
    try:
        # Check if the main function exists
        if hasattr(st_mermaid, 'st_mermaid'):
            MERMAID_FUNCTIONAL = True
        else:
            MERMAID_FUNCTIONAL = False
    except:
        MERMAID_FUNCTIONAL = False
except ImportError:
    MERMAID_AVAILABLE = False
    MERMAID_FUNCTIONAL = False

def clean_mermaid_diagram(diagram_code: str) -> str:
    """
    Clean and validate Mermaid diagram code for better Safari compatibility
    """
    if not diagram_code:
        return ""
    
    # Remove any problematic characters
    cleaned = diagram_code.strip()
    
    # Fix common syntax issues
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Remove empty lines
    
    # Ensure proper line endings
    lines = cleaned.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Skip empty lines
            if not line:
                continue
            
            # Format graph declaration
            if line.startswith('graph') or line.startswith('flowchart'):
                formatted_lines.append(line)
            else:
                # Indent all other lines
                formatted_lines.append('    ' + line)
    
    # Validate the diagram has basic structure
    result = '\n'.join(formatted_lines)
    
    # Ensure it starts with graph declaration
    if not result.startswith('graph') and not result.startswith('flowchart'):
        result = 'graph TD\n' + result
    
    return result

def detect_safari_browser() -> bool:
    """
    Detect if the user is using Safari browser
    """
    try:
        # Check session state for user agent
        user_agent = st.session_state.get('user_agent', '')
        if 'Safari' in user_agent and 'Chrome' not in user_agent:
            return True
        
        # Fallback: assume Safari if running on macOS and no Chrome detected
        import platform
        if platform.system() == 'Darwin':
            return True
            
    except Exception:
        pass
    
    return False

def simplify_diagram_for_safari(diagram_code: str) -> str:
    """
    Simplify Mermaid diagram for better Safari compatibility
    """
    if not diagram_code:
        return ""
    
    lines = diagram_code.split('\n')
    simplified_lines = []
    node_count = 0
    
    for line in lines:
        line = line.strip()
        if line:
            # Keep graph declaration
            if line.startswith('graph') or line.startswith('flowchart'):
                simplified_lines.append(line)
                continue
            
            # Limit the number of nodes to prevent Safari overload
            if node_count > 10:  # Even more conservative for Safari
                if '-->' in line:
                    continue
            
            # Remove styling that can cause Safari issues
            if line.startswith('style ') or line.startswith('classDef'):
                continue
            
            # Simplify node labels
            if '[' in line and ']' in line and '-->' not in line:
                # Shorten long labels
                start = line.find('[')
                end = line.find(']')
                if start < end:
                    label = line[start+1:end]
                    # Remove quotes and shorten
                    label = label.replace('"', '').replace("'", '')
                    if len(label) > 20:
                        label = label[:17] + "..."
                    line = line[:start+1] + f'"{label}"' + line[end:]
                node_count += 1
            
            simplified_lines.append(line)
    
    return '\n'.join(simplified_lines)

def create_plotly_diagram_from_mermaid(mermaid_code: str, title: str) -> Optional[go.Figure]:
    """
    Create a Plotly diagram from Mermaid code as a fallback
    """
    try:
        # Parse the Mermaid code to extract nodes and connections
        nodes = []
        edges = []
        node_positions = {}
        
        lines = mermaid_code.split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract node definitions
            if '[' in line and ']' in line and '-->' not in line:
                # Extract node ID and label
                parts = line.split('[')
                if len(parts) >= 2:
                    node_id = parts[0].strip()
                    label = parts[1].split(']')[0].strip().replace('"', '')
                    nodes.append({'id': node_id, 'label': label})
            
            # Extract connections
            if '-->' in line:
                parts = line.split('-->')
                if len(parts) == 2:
                    from_node = parts[0].strip()
                    to_node = parts[1].strip()
                    edges.append({'from': from_node, 'to': to_node})
        
        if not nodes:
            return None
        
        # Create positions for nodes (simple layout)
        import math
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / len(nodes)
            x = math.cos(angle)
            y = math.sin(angle)
            node_positions[node['id']] = (x, y)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge in edges:
            if edge['from'] in node_positions and edge['to'] in node_positions:
                x_from, y_from = node_positions[edge['from']]
                x_to, y_to = node_positions[edge['to']]
                
                fig.add_trace(go.Scatter(
                    x=[x_from, x_to, None],
                    y=[y_from, y_to, None],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add nodes
        x_coords = []
        y_coords = []
        labels = []
        
        for node in nodes:
            if node['id'] in node_positions:
                x, y = node_positions[node['id']]
                x_coords.append(x)
                y_coords.append(y)
                labels.append(node['label'])
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(size=30, color='lightblue', line=dict(width=2, color='darkblue')),
            text=labels,
            textposition='middle center',
            showlegend=False,
            hoverinfo='text'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Plotly diagram: {str(e)}")
        return None

def create_text_flowchart(diagram_code: str, title: str):
    """
    Create a simple text-based flowchart as fallback
    """
    st.markdown("#### 📊 Text-based Flow Visualization")
    
    if not diagram_code:
        st.info("No flow data available")
        return
    
    # Extract nodes and connections from Mermaid code
    nodes = []
    connections = []
    
    lines = diagram_code.split('\n')
    for line in lines:
        line = line.strip()
        if '-->' in line:
            parts = line.split('-->')
            if len(parts) == 2:
                from_node = parts[0].strip()
                to_node = parts[1].strip()
                
                # Clean node names
                from_node = re.sub(r'["\[\]{}]', '', from_node)
                to_node = re.sub(r'["\[\]{}]', '', to_node)
                
                connections.append((from_node, to_node))
                if from_node not in nodes:
                    nodes.append(from_node)
                if to_node not in nodes:
                    nodes.append(to_node)
    
    if connections:
        st.markdown("**Flow Steps:**")
        for i, (from_node, to_node) in enumerate(connections[:10], 1):
            st.markdown(f"{i}. `{from_node}` → `{to_node}`")
        
        if len(connections) > 10:
            st.markdown(f"... and {len(connections) - 10} more steps")
    else:
        st.info("Could not parse flow information from diagram")

def render_mermaid_diagram(diagram_code: str, title: str, description: str, height: int = 400, key_suffix: str = ""):
    """
    Render a Mermaid diagram with proper error handling and Safari compatibility
    
    Args:
        diagram_code: The Mermaid diagram code
        title: Title for the diagram section
        description: Description of what the diagram shows
        height: Height of the diagram in pixels
        key_suffix: Suffix for the unique key (to avoid conflicts)
    """
    st.markdown(f"#### {title}")
    st.markdown(description)
    
    # Check if diagram has meaningful content
    if not diagram_code or diagram_code.strip() == "graph TD\n    A[\"No main flows detected\"]":
        st.info("No diagram data available for this section.")
        return
    
    # Clean and validate diagram code
    cleaned_code = clean_mermaid_diagram(diagram_code)
    
    # Detect browser type for Safari-specific handling
    is_safari = detect_safari_browser()
    
    # Debug information
    with st.expander("🔧 Debug Info", expanded=False):
        st.write(f"**Mermaid Available:** {MERMAID_AVAILABLE}")
        st.write(f"**Mermaid Functional:** {MERMAID_FUNCTIONAL}")
        st.write(f"**Diagram Length:** {len(cleaned_code) if cleaned_code else 0}")
        st.write(f"**Key Suffix:** {key_suffix}")
        st.write(f"**Browser:** {'Safari' if is_safari else 'Other'}")
        if cleaned_code:
            st.write(f"**First 100 chars:** {cleaned_code[:100]}...")
    
    # Try multiple rendering approaches
    diagram_rendered = False
    
    if MERMAID_AVAILABLE and MERMAID_FUNCTIONAL:
        try:
            # Create a unique key for this diagram
            unique_key = f"mermaid_{key_suffix}_{abs(hash(cleaned_code)) % 10000}"
            
            # Safari-specific rendering with additional configuration
            if is_safari:
                # Use simpler diagram for Safari
                simplified_code = simplify_diagram_for_safari(cleaned_code)
                st_mermaid.st_mermaid(
                    simplified_code, 
                    height=height, 
                    key=unique_key
                )
            else:
                # Standard rendering for other browsers
                st_mermaid.st_mermaid(
                    cleaned_code, 
                    height=height, 
                    key=unique_key
                )
            diagram_rendered = True
            
        except Exception as e:
            st.warning(f"⚠️ Mermaid rendering failed: {str(e)}")
            # Continue to fallback options
    
    # Fallback 1: Create Plotly-based visualization
    if not diagram_rendered:
        st.info("🔄 Using alternative visualization (Safari-friendly)")
        
        # Create a Plotly-based diagram as fallback
        plotly_fig = create_plotly_diagram_from_mermaid(cleaned_code, title)
        if plotly_fig:
            st.plotly_chart(plotly_fig, use_container_width=True)
            diagram_rendered = True
        else:
            # Fallback to text-based flow chart
            create_text_flowchart(cleaned_code, title)
    
    # Always show expandable source code
    with st.expander("🔍 View Diagram Source Code"):
        st.code(cleaned_code, language='mermaid')
        st.markdown("**💡 Tip:** Copy this code to [Mermaid Live](https://mermaid.live/) if diagrams don't render")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create a direct link to view the diagram
            try:
                import urllib.parse
                
                # Create a link for Mermaid Live Editor
                encoded_diagram = urllib.parse.quote(diagram_code)
                live_editor_url = f"https://mermaid.live/edit#{encoded_diagram}"
                
                st.markdown(f"🎯 [**View Live Diagram**]({live_editor_url})")
                
            except Exception:
                pass  # If encoding fails, just skip the live link
        
        with col2:
            st.markdown("**Quick Links:**")
            st.markdown("🔗 [Mermaid Live Editor](https://mermaid.live)")
            st.markdown("📚 [Mermaid Documentation](https://mermaid.js.org/)")
        
        # Create alternative text-based visualization
        _create_text_based_diagram(diagram_code, title)

def _create_text_based_diagram(diagram_code: str, title: str):
    """Create a simple text-based representation of the diagram"""
    try:
        st.markdown("### 🎨 Text-Based Visualization")
        
        lines = diagram_code.split('\n')
        nodes = []
        connections = []
        
        for line in lines:
            line = line.strip()
            if '-->' in line:
                parts = line.split('-->')
                if len(parts) == 2:
                    from_node = parts[0].strip()
                    to_node = parts[1].strip()
                    connections.append(f"**{from_node}** → **{to_node}**")
            elif '[' in line and ']' in line:
                # Extract node name
                start = line.find('[')
                end = line.find(']')
                if start < end:
                    node_text = line[start+1:end].replace('"', '')
                    nodes.append(f"📦 {node_text}")
        
        if nodes:
            st.markdown("**Components:**")
            for node in nodes[:10]:  # Show first 10
                st.markdown(f"- {node}")
        
        if connections:
            st.markdown("**Flow:**")
            for conn in connections[:10]:  # Show first 10
                st.markdown(f"- {conn}")
                
    except Exception:
        st.info("Could not create text visualization")

def create_plotly_network_diagram(visual_data: Dict[str, Any], title: str) -> Optional[go.Figure]:
    """
    Create a Plotly-based network diagram from visual data
    """
    try:
        component_interactions = visual_data.get('component_interactions', [])
        
        if not component_interactions:
            return None
        
        # Extract unique components
        components = set()
        edges = []
        
        for interaction in component_interactions:
            from_comp = interaction.get('from_component', 'Unknown')
            to_comp = interaction.get('to_component', 'Unknown')
            components.add(from_comp)
            components.add(to_comp)
            edges.append((from_comp, to_comp))
        
        components = list(components)
        
        # Create positions for components (circular layout)
        import math
        positions = {}
        for i, comp in enumerate(components):
            angle = 2 * math.pi * i / len(components)
            x = math.cos(angle)
            y = math.sin(angle)
            positions[comp] = (x, y)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        for from_comp, to_comp in edges:
            if from_comp in positions and to_comp in positions:
                x_from, y_from = positions[from_comp]
                x_to, y_to = positions[to_comp]
                
                fig.add_trace(go.Scatter(
                    x=[x_from, x_to, None],
                    y=[y_from, y_to, None],
                    mode='lines',
                    line=dict(color='rgba(50, 50, 50, 0.5)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add nodes
        x_coords = []
        y_coords = []
        labels = []
        
        for comp in components:
            x, y = positions[comp]
            x_coords.append(x)
            y_coords.append(y)
            labels.append(comp[:20])  # Truncate long names
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(size=40, color='lightblue', line=dict(width=2, color='navy')),
            text=labels,
            textposition='middle center',
            textfont=dict(size=10),
            showlegend=False,
            hoverinfo='text'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating network diagram: {str(e)}")
        return None


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
Analyze this file in detail.

{context}

Provide a comprehensive explanation (at least 5-7 sentences, 200-400 words) covering:
1. What this file does overall and its role in the project (e.g., is it a core model, utility, or config?).
2. Key functions/classes: For each (up to 5), describe what it does, its parameters, return values, how it works (with pseudocode/examples), and edge cases (e.g., what if input is empty?).
3. Interactions: How it uses imports or is used by other files.
4. Strengths/weaknesses: Any best practices or potential issues.
Use clear, descriptive language with examples where helpful.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a detailed code analyst explaining files thoroughly."},
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

Please provide in MARKDOWN format:
1. **Overall Assessment** - What type of project is this (e.g., ML transformer training)? Scope, key goals, and tech stack details.
2. **Code Quality Observations** - Analyze organization, quality metrics (e.g., why low duplication?), strengths (e.g., modular design), and weaknesses (e.g., missing tests). Include examples.
3. **Key Recommendations** - 3-5 specific improvements (e.g., add unit tests for models). Explain WHY each helps.
Be insightful, reference specific files/metrics, and aim for 300-500 words.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a senior ML architect analyzing a codebase including quality metrics."},
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
    
    # Mermaid status
    if MERMAID_AVAILABLE and MERMAID_FUNCTIONAL:
        st.markdown('<div class="status-success">✅ Mermaid diagrams enabled</div>', unsafe_allow_html=True)
        
        # Test Mermaid rendering
        if st.button("🧪 Test Mermaid"):
            st.markdown("**Mermaid Test:**")
            try:
                test_diagram = "graph TD\n    A[Test] --> B[Working!]"
                st_mermaid.st_mermaid(test_diagram, height=150, key="sidebar_test")
                st.success("Mermaid is working!")
            except Exception as e:
                st.error(f"Mermaid test failed: {str(e)}")
    elif MERMAID_AVAILABLE:
        st.markdown('<div class="status-error">⚠️ Mermaid installed but not functional</div>', unsafe_allow_html=True)
        st.markdown("""
        **Try fixing:**
        1. Restart Streamlit
        2. `pip uninstall streamlit-mermaid && pip install streamlit-mermaid`
        """)
    else:
        st.markdown('<div class="status-error">❌ Mermaid not available</div>', unsafe_allow_html=True)
        st.markdown("""
        **To enable interactive diagrams:**
        1. `pip install streamlit-mermaid`
        2. Restart the app
        """)
        st.info("You can still view diagram code and use Mermaid Live Editor")
    
    
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

detail_level = st.radio("Explanation Detail", ["Concise", "Detailed"], index=1)

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
            
            # Debug information
            with st.expander("🔍 Debug: Visual Analysis Data", expanded=True):
                st.write("**Quality Metrics Keys:**", list(quality_metrics.keys()) if quality_metrics else "No quality metrics")
                st.write("**Visual Data Keys:**", list(visual_data.keys()) if visual_data else "No visual data")
                st.write("**Visual Data:**", visual_data)
                
                if visual_data:
                    st.write("**Entry Points:**", visual_data.get('entry_points', 'No entry points'))
                    st.write("**Has Entry Points:**", bool(visual_data.get('entry_points')))
                    st.write("**Visual Diagrams:**", visual_data.get('visual_diagrams', 'No visual diagrams'))
            
            if visual_data and visual_data.get('visual_diagrams', {}):
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
                
                for diagram_name, diagram_code in visual_diagrams.items():
                    if diagram_code and diagram_code.strip():
                        title = f"{diagram_name.replace('_', ' ').title()}"
                        description = "Diagram showing code structure"
                        
                        if detect_safari_browser():
                            plotly_fig = create_plotly_diagram_from_mermaid(diagram_code, title)
                            if plotly_fig:
                                st.plotly_chart(plotly_fig)
                            else:
                                st.code(diagram_code, language='mermaid')
                        else:
                            render_mermaid_diagram(diagram_code, title, description)
                
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
                st.warning("⚠️ No visual data detected—repo may lack supported structures.")
                st.markdown("**Possible reasons:**")
                st.markdown("- No entry points detected in the code")
                st.markdown("- Visual analyzer failed to process the files")
                st.markdown("- Repository has no supported file types (.py, .js, .ts, etc.)")
                
                # Show what we can
                if visual_data:
                    st.markdown("**Available data:**")
                    for key, value in visual_data.items():
                        if value:
                            st.markdown(f"- **{key}:** {len(value) if isinstance(value, (list, dict)) else value}")
                
                # Force show diagrams if they exist
                visual_diagrams = visual_data.get('visual_diagrams', {}) if visual_data else {}
                if visual_diagrams:
                    st.markdown("### 🔧 Available Diagrams (Debug)")
                    for diagram_name, diagram_code in visual_diagrams.items():
                        if diagram_code and diagram_code.strip():
                            st.markdown(f"**{diagram_name}:**")
                            st.code(diagram_code, language='text')
                
                st.info("📊 Try analyzing a Python repository with main() functions or __main__ blocks for better results.")
    else:
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")