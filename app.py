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
from visual_code_analyzer import IntelligentAdaptiveVisualAnalyzer

# Import new analyzers
from code_evolution_analyzer import CodeEvolutionAnalyzer
from code_mentor import InteractiveCodeMentor
from performance_predictor import PerformancePredictor

# Import Google Gemini for hybrid analysis
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
    """Enhanced repository analyzer with hybrid Gemini/Ollama approach"""
    
    def __init__(self, gemini_api_key: str = None):
        # Initialize Gemini if API key is provided
        self.gemini_model = None
        self.gemini_available = False
        self.api_requests_made = 0
        self.api_limit_reached = False
        
        if gemini_api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_available = True
            except Exception as e:
                st.warning(f"⚠️ Gemini API initialization failed: {str(e)}")
                self.gemini_available = False
        
        # Initialize Ollama
        self.ollama_available = self._check_ollama()
        
        # Initialize analyzers
        self.quality_analyzer = CodeQualityAnalyzer()
        self.visual_analyzer = IntelligentAdaptiveVisualAnalyzer()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    async def _call_gemini(self, messages: List[Dict], max_retries: int = 3) -> str:
        """Call Gemini API with retry logic"""
        if not self.gemini_available or self.api_limit_reached:
            return None
        
        # Convert messages to a single prompt
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
        
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(prompt)
                if response.text:
                    self.api_requests_made += 1
                    return response.text.strip()
                else:
                    return None
            except Exception as e:
                error_msg = str(e).lower()
                if 'quota' in error_msg or 'rate limit' in error_msg:
                    self.api_limit_reached = True
                    return None
                elif attempt == max_retries - 1:
                    return None
                else:
                    # Wait before retry
                    await asyncio.sleep(2 ** attempt)
        
        return None
    
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
    
    async def _call_llm_hybrid(self, messages: List[Dict], context: str = "analysis") -> str:
        """Hybrid LLM call: Try Gemini first, fallback to Ollama"""
        
        # Try Gemini first if available and not over quota
        if self.gemini_available and not self.api_limit_reached:
            try:
                response = await self._call_gemini(messages)
                if response:
                    return response
                else:
                    # If Gemini fails, log it and fallback to Ollama
                    if self.api_limit_reached:
                        st.info("🔄 API quota reached, switching to local Ollama...")
                    else:
                        st.info("🔄 Gemini API failed, switching to local Ollama...")
            except Exception as e:
                st.info(f"🔄 Gemini error: {str(e)}, switching to local Ollama...")
        
        # Fallback to Ollama
        if self.ollama_available:
            return await self._call_ollama(messages)
        else:
            return f"Error: Neither Gemini API nor Ollama is available for {context}"
    
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
        if not self.gemini_available and not self.ollama_available:
            return f"File analysis requires either Gemini API or Ollama to be running. File has {file_info.get('lines', 0)} lines."
        
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
            
            response = await self._call_llm_hybrid(messages, "file analysis")
            return response.strip()
            
        except Exception as e:
            return f"File analysis failed: {str(e)}"
    
    async def _generate_file_explanations(self, detailed_files: Dict[str, Any]) -> Dict[str, str]:
        """Generate explanations for individual files using LLM"""
        if not self.gemini_available and not self.ollama_available:
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
        if not self.gemini_available and not self.ollama_available:
            return "Overall analysis requires either Gemini API or Ollama to be running."
        
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
            
            response = await self._call_llm_hybrid(messages, "overall analysis")
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get current analyzer status"""
        return {
            "gemini_available": self.gemini_available,
            "ollama_available": self.ollama_available,
            "api_requests_made": self.api_requests_made,
            "api_limit_reached": self.api_limit_reached,
            "recommended_approach": self._get_recommended_approach(),
            "has_any_llm": self.gemini_available or self.ollama_available
        }
    
    def _get_recommended_approach(self) -> str:
        """Get recommendation based on current status"""
        if self.api_limit_reached:
            return "Local Ollama (API quota reached)"
        elif self.gemini_available:
            return "Gemini API (Fast)"
        elif self.ollama_available:
            return "Local Ollama (Slower)"
        else:
            return "Neither available"

# Main Streamlit App (Enhanced with Quality Metrics UI)
st.markdown('<h1 class="main-title">🚀 Hybrid LLM Repository Analyzer</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive GitHub repository analysis with quality metrics and interactive visual diagrams - powered by Gemini API with Ollama fallback!**")

# Initialize analyzer with hybrid approach
@st.cache_resource
def get_analyzer(gemini_api_key: str = None):
    return RepositoryAnalyzer(gemini_api_key=gemini_api_key)

# Get API key from sidebar
with st.sidebar:
    st.header("🚀 Hybrid Configuration")
    
    gemini_api_key = st.text_input(
        "🔑 Gemini API Key (Optional)",
        type="password",
        help="For fastest analysis. Leave empty to use local Ollama only.",
        placeholder="Enter your Gemini API key here..."
    )
    
    # Get analyzer with API key
    if gemini_api_key:
        analyzer = get_analyzer(gemini_api_key)
    else:
        analyzer = get_analyzer()
    
    # Show status
    status = analyzer.get_status()
    
    st.subheader("📊 System Status")
    
    if status["gemini_available"]:
        st.success("✅ Gemini API ready (Fast)")
    else:
        st.info("⚪ Gemini API not configured")
    
    if status["ollama_available"]:
        st.success("✅ Ollama ready (Local)")
    else:
        st.error("❌ Ollama not available")
    
    if not status["has_any_llm"]:
        st.error("❌ No LLM available for analysis")
    else:
        st.info(f"**Current approach:** {status['recommended_approach']}")
    
    # Show API usage if applicable
    if status["api_requests_made"] > 0:
        st.metric("API Requests Used", status["api_requests_made"])
    
    # Show performance tips
    st.markdown("### ⚡ Performance Tips:")
    st.markdown("""
    **For Fastest Analysis:**
    - Use Gemini API key
    - Analyze smaller repositories first
    
    **For Ollama Optimization:**
    - Use `llama3.2:1b` for speed
    - Ensure Ollama is running
    - Close other applications
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
    status = analyzer.get_status()
    if not status["has_any_llm"]:
        st.error("❌ No LLM available for analysis. Please configure Gemini API key or install Ollama.")
        st.stop()
    elif not status["gemini_available"] and not status["ollama_available"]:
        st.warning("⚠️ No LLM available. Analysis will work but file explanations will be limited.")
    elif status["gemini_available"] and not status["api_limit_reached"]:
        st.info("🚀 Using Gemini API for fast analysis...")
    elif status["ollama_available"]:
        st.info("🤖 Using local Ollama for analysis...")
    
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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📁 File Structure", 
            "📄 File Explanations",
            "🤖 AI Insights",
            "⚙️ Technologies",
            "🏆 Code Quality",
            "🎨 Visual Analysis",
            "📈 Evolution Analysis",
            "🎓 Code Mentor",
            "⚡ Performance Prediction"
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
                st.info("File explanations not available. Configure Gemini API key or make sure Ollama is running and re-run the analysis.")
        
        with tab3:
            st.subheader("AI-Generated Insights")
            
            insights = result.get('insights', '')
            if insights:
                st.markdown(insights)
            else:
                st.info("AI insights not available. Configure Gemini API key or make sure Ollama is running.")
        
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
        
        # NEW: Enhanced Visual Analysis Tab
        with tab6:
            st.subheader("🎨 Intelligent Visual Analysis")
            
            visual_data = quality_metrics.get('visual_analysis', {})
            
            if visual_data and visual_data.get('codebase_profile', {}):
                # Display codebase profile
                codebase_profile = visual_data.get('codebase_profile', {})
                
                st.markdown("### 🔍 Codebase Profile")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    primary_lang = codebase_profile.get('primary_language', 'Unknown')
                    st.metric("Primary Language", primary_lang)
                
                with col2:
                    app_type = codebase_profile.get('application_type', 'Unknown')
                    st.metric("Application Type", app_type)
                
                with col3:
                    complexity = codebase_profile.get('complexity_indicators', {}).get('complexity_level', 'Unknown')
                    st.metric("Complexity Level", complexity)
                
                with col4:
                    frameworks = codebase_profile.get('detected_frameworks', [])
                    st.metric("Frameworks Found", len(frameworks))
                
                # Display detected frameworks
                if frameworks:
                    st.markdown("**🔧 Detected Frameworks:**")
                    framework_text = ", ".join(frameworks[:8])
                    st.info(framework_text)
                
                # Entry Points Analysis
                entry_points = visual_data.get('entry_points', [])
                if entry_points:
                    st.markdown("### 🎯 Intelligent Entry Point Detection")
                    
                    for ep in entry_points:
                        confidence = ep.get('confidence', 0)
                        confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                        
                        with st.expander(f"{confidence_color} {ep['name']} ({ep['type'].replace('_', ' ').title()})"):
                            st.markdown(f"**File:** `{ep['file']}`")
                            st.markdown(f"**Description:** {ep['description']}")
                            st.markdown(f"**Confidence:** {confidence:.2f}")
                            st.markdown(f"**Discovery Method:** {ep.get('discovery_method', 'pattern_detection')}")
                
                # Execution Flows Analysis
                execution_flows = visual_data.get('execution_flows', [])
                if execution_flows:
                    st.markdown("### 🌊 Adaptive Execution Flow Analysis")
                    
                    for i, flow in enumerate(execution_flows, 1):
                        entry_point = flow['entry_point']
                        flow_steps = flow['flow_steps']
                        
                        with st.expander(f"Flow {i}: {entry_point['name']} ({len(flow_steps)} steps)"):
                            st.markdown(f"**Application Type:** {app_type}")
                            st.markdown(f"**Execution Pattern:** {flow.get('execution_pattern', 'Unknown')}")
                            
                            if flow.get('discovered_operations'):
                                st.markdown("**Discovered Operations:**")
                                for op in flow['discovered_operations'][:5]:
                                    st.markdown(f"- {op}")
                            
                            # Show flow steps
                            st.markdown("**Execution Steps:**")
                            for step in flow_steps[:8]:
                                step_emoji = {
                                    'initialization': '🚀',
                                    'dependency_loading': '📦',
                                    'configuration': '⚙️',
                                    'operation': '⚡',
                                    'control_flow': '🔄',
                                    'entry': '🎯'
                                }.get(step.get('type'), '▶️')
                                
                                st.markdown(f"- {step_emoji} **{step['action']}** ({step.get('category', 'unknown')})")
                
                # Component Interactions
                component_interactions = visual_data.get('component_interactions', [])
                if component_interactions:
                    st.markdown("### 🔗 Intelligent Component Analysis")
                    st.markdown(f"Found {len(component_interactions)} component interactions:")
                    
                    # Group by strength
                    strong_interactions = [i for i in component_interactions if i.get('strength') == 'strong']
                    medium_interactions = [i for i in component_interactions if i.get('strength') == 'medium']
                    weak_interactions = [i for i in component_interactions if i.get('strength') == 'weak']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Strong Dependencies", len(strong_interactions))
                    with col2:
                        st.metric("Medium Dependencies", len(medium_interactions))
                    with col3:
                        st.metric("Weak Dependencies", len(weak_interactions))
                    
                    # Show top interactions
                    for interaction in component_interactions[:8]:
                        strength_emoji = {"strong": "🔴", "medium": "🟡", "weak": "🟢"}.get(interaction.get('strength'), "⚪")
                        st.markdown(f"{strength_emoji} **{interaction['from_component']}** → **{interaction['to_component']}** ({interaction.get('interaction_type', 'dependency')})")
                
                # Data Flow Analysis
                data_flows = visual_data.get('data_flows', [])
                if data_flows:
                    st.markdown("### 📈 Dynamic Data Flow Analysis")
                    
                    for flow in data_flows[:5]:
                        file_name = Path(flow['file']).name
                        complexity_score = flow.get('complexity', 0)
                        
                        with st.expander(f"📄 {file_name} (Complexity: {complexity_score})"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**📥 Inputs:**")
                                for inp in flow['inputs']:
                                    st.markdown(f"- {inp}")
                            
                            with col2:
                                st.markdown("**⚙️ Transformations:**")
                                for transform in flow['transformations']:
                                    st.markdown(f"- {transform}")
                            
                            with col3:
                                st.markdown("**📤 Outputs:**")
                                for output in flow['outputs']:
                                    st.markdown(f"- {output}")
                
                # Architectural Insights
                arch_insights = visual_data.get('architectural_insights', {})
                if arch_insights:
                    st.markdown("### 🏗️ Architectural Pattern Detection")
                    
                    arch_style = arch_insights.get('architectural_style', 'Unknown')
                    st.info(f"**Detected Architecture:** {arch_style}")
                    
                    detected_patterns = arch_insights.get('detected_patterns', [])
                    if detected_patterns:
                        st.markdown("**🎨 Detected Patterns:**")
                        for pattern in detected_patterns:
                            st.markdown(f"- {pattern}")
                
                # Visual Diagrams Section
                st.markdown("### 📊 Adaptive Visual Diagrams")
                
                visual_diagrams = visual_data.get('visual_diagrams', {})
                
                if visual_diagrams:
                    for diagram_name, diagram_code in visual_diagrams.items():
                        if diagram_code and diagram_code.strip():
                            title = f"{diagram_name.replace('_', ' ').title()}"
                            description = f"Adaptive diagram showing {diagram_name.replace('_', ' ')}"
                            
                            render_mermaid_diagram(diagram_code, title, description, key_suffix=diagram_name)
                else:
                    st.info("No visual diagrams generated - this may indicate a very simple codebase structure")
                
                # Intelligent Insights
                intelligent_insights = visual_data.get('intelligent_insights', [])
                if intelligent_insights:
                    st.markdown("### 💡 Intelligent Insights")
                    for insight in intelligent_insights:
                        st.markdown(f"- {insight}")
                
                # Adaptive Recommendations
                adaptive_recommendations = visual_data.get('adaptive_recommendations', [])
                if adaptive_recommendations:
                    st.markdown("### 🎯 Adaptive Recommendations")
                    for rec in adaptive_recommendations:
                        st.markdown(f'''
                        <div class="recommendation-box">
                        {rec}
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Analysis Strategy
                analysis_strategy = visual_data.get('analysis_strategy', {})
                if analysis_strategy:
                    with st.expander("🔧 Analysis Strategy Used"):
                        st.json(analysis_strategy)
                        st.markdown(f"**Approach:** {analysis_strategy.get('approach', 'Unknown')}")
                        st.markdown(f"**Depth:** {analysis_strategy.get('depth', 'Unknown')}")
                        st.markdown(f"**Focus Areas:** {', '.join(analysis_strategy.get('focus_areas', []))}")
            
            else:
                st.warning("⚠️ No intelligent visual analysis data available")
                st.markdown("**Possible reasons:**")
                st.markdown("- Analysis failed to complete")
                st.markdown("- Codebase structure not detected")
                st.markdown("- No supported file types found")
                
                st.info("💡 The intelligent analyzer works best with Python, JavaScript, TypeScript, Java, and other common programming languages.")
        
        # NEW TAB 7: Evolution Analysis
        with tab7:
            st.subheader("📈 Code Evolution Analysis")
            
            if st.button("🕰️ Analyze Repository Evolution", key="evolution_button"):
                with st.spinner("Analyzing repository evolution... This may take a few minutes"):
                    try:
                        # Initialize evolution analyzer
                        evolution_analyzer = CodeEvolutionAnalyzer(repo_path)
                        
                        # Run the analysis
                        evolution_report = evolution_analyzer.analyze_evolution(months_back=6)
                        
                        # Display results
                        st.success("Evolution analysis complete!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Functions Tracked", 
                                     evolution_report['summary']['functions_tracked'])
                        with col2:
                            st.metric("Rapid Growth", 
                                     evolution_report['summary']['rapidly_growing_functions'],
                                     delta="functions")
                        with col3:
                            st.metric("Bug-Prone Files", 
                                     evolution_report['summary']['bug_prone_files'])
                        with col4:
                            st.metric("Refactoring Needed", 
                                     evolution_report['summary']['refactoring_needed'])
                        
                        # Growth Alerts
                        if evolution_report['growth_alerts']:
                            st.markdown("### 📊 Growth Alerts")
                            for alert in evolution_report['growth_alerts'][:5]:
                                with st.expander(f"⚠️ {alert['function']}"):
                                    if alert['type'] == 'rapid_growth':
                                        st.write(f"**Growth Rate:** {alert['growth_rate']:.0f}%")
                                        st.write(f"**Lines:** {alert['start_lines']} → {alert['end_lines']}")
                                    elif alert['type'] == 'complexity_growth':
                                        st.write(f"**Complexity:** {alert['start_complexity']} → {alert['end_complexity']}")
                                    st.info(alert['recommendation'])
                        
                        # Bug Patterns
                        if evolution_report['bug_patterns']:
                            st.markdown("### 🐛 Bug-Prone Files")
                            for pattern in evolution_report['bug_patterns'][:5]:
                                st.warning(f"**{pattern['file']}** - {pattern['bug_fixes']} bug fixes")
                                st.caption(pattern['recommendation'])
                        
                        # Refactoring Predictions
                        if evolution_report['refactoring_predictions']:
                            st.markdown("### 🔮 Refactoring Predictions")
                            for prediction in evolution_report['refactoring_predictions'][:5]:
                                urgency = "🔴" if prediction['priority'] == 'high' else "🟡"
                                st.write(f"{urgency} **{prediction['function']}**")
                                st.write(f"Current: {prediction['current_lines']} lines → "
                                       f"Predicted: {prediction['predicted_lines']} lines")
                                st.info(prediction['recommendation'])
                        
                        # Insights
                        st.markdown("### 💡 Key Insights")
                        for insight in evolution_report['insights']:
                            st.write(insight)
                            
                        # Visualizations
                        if evolution_report['timeline_data']['file_growth']:
                            st.markdown("### 📈 File Growth Over Time")
                            
                            # Create plotly chart
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            for file_data in evolution_report['timeline_data']['file_growth'][:5]:
                                dates = [point[0] for point in file_data['data_points']]
                                lines = [point[1] for point in file_data['data_points']]
                                
                                fig.add_trace(go.Scatter(
                                    x=dates, y=lines,
                                    mode='lines+markers',
                                    name=file_data['file']
                                ))
                            
                            fig.update_layout(
                                title="File Size Evolution",
                                xaxis_title="Date",
                                yaxis_title="Lines of Code"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Evolution analysis failed: {str(e)}")
        
        # NEW TAB 8: Code Mentor
        with tab8:
            st.subheader("🎓 Interactive Code Mentor")
            
            # Initialize mentor in session state
            if 'code_mentor' not in st.session_state:
                if 'analysis_results' in st.session_state:
                    st.session_state.code_mentor = InteractiveCodeMentor(
                        st.session_state['analysis_results'],
                        analyzer._call_llm_hybrid  # Your hybrid LLM function
                    )
                    # Start the session
                    welcome = st.session_state.code_mentor.start_mentor_session()
                    st.session_state.mentor_messages = [welcome]
                else:
                    st.warning("Please analyze a repository first to start the Code Mentor.")
                    st.stop()
            
            # Display conversation history
            for message in st.session_state.get('mentor_messages', []):
                if message.get('message'):
                    with st.chat_message("assistant" if 'suggestions' in message else "user"):
                        st.write(message['message'])
                        
                        # Show diagram if available
                        if message.get('diagram'):
                            render_mermaid_diagram(
                                message['diagram'],
                                "Explanation Diagram",
                                "Visual representation of the concept",
                                key_suffix=f"mentor_{len(st.session_state.mentor_messages)}"
                            )
                        
                        # Show code examples if available
                        if message.get('code_examples'):
                            st.markdown("**📝 Relevant Code Examples:**")
                            for example in message['code_examples']:
                                st.code(f"# {example['description']}\n# See: {example['file']}")
                        
                        # Show exercise if created
                        if message.get('exercise'):
                            exercise = message['exercise']
                            with st.expander("📚 Exercise Details"):
                                st.write(exercise['content'])
                                
                                # Setup commands
                                if message.get('setup_commands'):
                                    st.markdown("**Setup Commands:**")
                                    st.code('\n'.join(message['setup_commands']))
                                
                                # Hint system
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("💡 Hint 1", key=f"hint1_{exercise['id']}"):
                                        hint = asyncio.run(
                                            st.session_state.code_mentor.provide_hint(exercise['id'], 1)
                                        )
                                        st.info(hint)
                                with col2:
                                    if st.button("💡 Hint 2", key=f"hint2_{exercise['id']}"):
                                        hint = asyncio.run(
                                            st.session_state.code_mentor.provide_hint(exercise['id'], 2)
                                        )
                                        st.info(hint)
                                with col3:
                                    if st.button("💡 Hint 3", key=f"hint3_{exercise['id']}"):
                                        hint = asyncio.run(
                                            st.session_state.code_mentor.provide_hint(exercise['id'], 3)
                                        )
                                        st.info(hint)
                                
                                # Solution submission
                                solution_code = st.text_area(
                                    "Your Solution:",
                                    key=f"solution_{exercise['id']}",
                                    height=200
                                )
                                
                                if st.button("Check Solution", key=f"check_{exercise['id']}"):
                                    if solution_code:
                                        result = asyncio.run(
                                            st.session_state.code_mentor.check_solution(
                                                exercise['id'], solution_code
                                            )
                                        )
                                        if result['success']:
                                            st.success("Solution submitted!")
                                            st.write(result['feedback'])
                                            if result['exercise_completed']:
                                                st.balloons()
                                                st.success("🎉 Exercise completed successfully!")
                                            
                                            # Next steps
                                            if result['next_steps']:
                                                st.markdown("**Next Steps:**")
                                                for step in result['next_steps']:
                                                    st.write(f"• {step}")
            
            # Suggestions
            if st.session_state.get('mentor_messages'):
                last_message = st.session_state.mentor_messages[-1]
                if last_message.get('follow_up_suggestions') or last_message.get('suggestions'):
                    st.markdown("**💭 Suggested Questions:**")
                    suggestions = (last_message.get('follow_up_suggestions') or 
                                 last_message.get('suggestions', []))
                    
                    cols = st.columns(min(len(suggestions), 3))
                    for i, suggestion in enumerate(suggestions):
                        with cols[i % 3]:
                            if st.button(suggestion, key=f"suggestion_{i}"):
                                st.session_state.pending_question = suggestion
            
            # Chat input
            question = st.chat_input("Ask me anything about this codebase...")
            
            # Handle pending question from suggestion button
            if 'pending_question' in st.session_state:
                question = st.session_state.pending_question
                del st.session_state.pending_question
            
            if question:
                # Add user message to display
                st.session_state.mentor_messages.append({'message': question})
                
                # Process the question
                with st.spinner("Thinking..."):
                    response = asyncio.run(
                        st.session_state.code_mentor.process_question(question)
                    )
                    st.session_state.mentor_messages.append(response)
                
                # Rerun to display new messages
                st.rerun()
            
            # Learning progress sidebar
            with st.sidebar:
                st.markdown("### 📊 Learning Progress")
                
                mentor = st.session_state.get('code_mentor')
                if mentor:
                    context = mentor.learning_context
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Files Explored", len(context['explored_files']))
                    with col2:
                        st.metric("Exercises", len(context['generated_exercises']))
                    
                    # Progress bar
                    progress = min(len(context['explored_files']) * 10, 100)
                    st.progress(progress / 100)
                    st.caption(f"Understanding: {context['understanding_level'].title()}")
                    
                    # Completed exercises
                    completed = [e for e in context['generated_exercises'] if e.get('completed')]
                    if completed:
                        st.markdown("**✅ Completed Exercises:**")
                        for exercise in completed:
                            st.write(f"• {exercise['topic']} ({exercise['difficulty']})")
        
        # NEW TAB 9: Performance Prediction
        with tab9:
            st.subheader("⚡ Performance Prediction Engine")
            
            # Get analysis data
            if 'analysis_results' not in st.session_state:
                st.warning("Please analyze a repository first to predict performance.")
                st.stop()
            
            detailed_files = st.session_state['analysis_results']['analysis']['detailed_files']
            
            # Performance analysis button
            if st.button("🔍 Analyze Performance", key="perf_button"):
                with st.spinner("Predicting performance bottlenecks..."):
                    # Initialize performance predictor
                    predictor = PerformancePredictor()
                    
                    # Run analysis
                    perf_report = predictor.analyze_performance(detailed_files)
                    
                    # Store in session state
                    st.session_state['performance_report'] = perf_report
            
            # Display results if available
            if 'performance_report' in st.session_state:
                report = st.session_state['performance_report']
                
                # Performance Score Card
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Display score with color coding
                    score = report['score']
                    grade = report['grade']
                    
                    score_color = {
                        'A': '#4CAF50',
                        'B': '#8BC34A', 
                        'C': '#FFC107',
                        'D': '#FF9800',
                        'F': '#F44336'
                    }.get(grade, '#757575')
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: {score_color}; 
                                color: white; border-radius: 10px;">
                        <h1 style="margin: 0;">{score}/100</h1>
                        <h3 style="margin: 0;">Grade: {grade}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Total Issues", report['summary']['total_issues'])
                    st.metric("Critical Issues", report['summary']['critical_issues'], 
                             delta="-" + str(report['summary']['critical_issues']) if report['summary']['critical_issues'] > 0 else "0",
                             delta_color="inverse")
                
                with col3:
                    st.metric("Files Analyzed", report['summary']['files_analyzed'])
                    st.metric("Functions Analyzed", report['summary']['functions_analyzed'])
                
                with col4:
                    # Issue breakdown pie chart
                    severity_counts = {
                        'Critical': report['summary']['critical_issues'],
                        'High': report['summary']['high_issues'],
                        'Medium': len(report['issues_by_severity'].get('medium', [])),
                        'Low': len(report['issues_by_severity'].get('low', []))
                    }
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=list(severity_counts.keys()),
                        values=list(severity_counts.values()),
                        hole=.3,
                        marker_colors=['#F44336', '#FF9800', '#FFC107', '#4CAF50']
                    )])
                    fig.update_layout(
                        showlegend=False,
                        height=200,
                        margin=dict(t=0, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance Insights
                st.markdown("### 💡 Performance Insights")
                for insight in report['insights']:
                    st.write(insight)
                
                # Critical Issues
                if report['issues_by_severity'].get('critical'):
                    st.markdown("### 🚨 Critical Performance Issues")
                    for issue in report['issues_by_severity']['critical'][:5]:
                        with st.expander(f"🔴 {issue.description}"):
                            st.write(f"**Location:** `{issue.location}`")
                            st.write(f"**Impact:** {issue.impact}")
                            st.write(f"**Suggestion:** {issue.suggestion}")
                            
                            if issue.code_snippet:
                                st.code(issue.code_snippet, language='python')
                            
                            if issue.estimated_complexity:
                                st.write(f"**Complexity:** {issue.estimated_complexity}")
                
                # High Priority Issues
                if report['issues_by_severity'].get('high'):
                    st.markdown("### ⚠️ High Priority Issues")
                    for issue in report['issues_by_severity']['high'][:5]:
                        with st.expander(f"🟡 {issue.description}"):
                            st.write(f"**Location:** `{issue.location}`")
                            st.write(f"**Impact:** {issue.impact}")
                            st.write(f"**Suggestion:** {issue.suggestion}")
                            
                            if issue.code_snippet:
                                st.code(issue.code_snippet, language='python')
                
                # Performance Roadmap
                if report.get('performance_roadmap'):
                    st.markdown("### 🗺️ Performance Improvement Roadmap")
                    for phase in report['performance_roadmap']:
                        with st.expander(f"📋 {phase['phase']} ({phase['timeline']})"):
                            st.write(phase['description'])
                            for item in phase['items']:
                                st.write(f"• **{item['title']}** - {item['description']}")
                                st.caption(f"Impact: {item['impact']} | Effort: {item['effort']}")
    else:
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")