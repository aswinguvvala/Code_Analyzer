# Enhanced app.py - Repository analyzer with code quality metrics

import streamlit as st
import asyncio
import json
import time
import os
import tempfile
import shutil
import re
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import git
import concurrent.futures
from functools import partial
import hashlib
import io

# Import our code quality analyzer
from code_quality_analyzer import CodeQualityAnalyzer

# Import new analyzers
from code_mentor import InteractiveCodeMentor

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
    st.markdown("#### Text-based Flow Visualization")
    
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
    with st.expander("Debug Info", expanded=False):
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
            st.warning(f"Warning: Mermaid rendering failed: {str(e)}")
            # Continue to fallback options
    
    # Fallback 1: Create Plotly-based visualization
    if not diagram_rendered:
        st.info("Using alternative visualization (Safari-friendly)")
        
        # Create a Plotly-based diagram as fallback
        plotly_fig = create_plotly_diagram_from_mermaid(cleaned_code, title)
        if plotly_fig:
            st.plotly_chart(plotly_fig, use_container_width=True)
            diagram_rendered = True
        else:
            # Fallback to text-based flow chart
            create_text_flowchart(cleaned_code, title)
    
    # Always show expandable source code
    with st.expander("View Diagram Source Code"):
        st.code(cleaned_code, language='mermaid')
        st.markdown("**Tip:** Copy this code to [Mermaid Live](https://mermaid.live/) if diagrams don't render")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create a direct link to view the diagram
            try:
                import urllib.parse
                
                # Create a link for Mermaid Live Editor
                encoded_diagram = urllib.parse.quote(diagram_code)
                live_editor_url = f"https://mermaid.live/edit#{encoded_diagram}"
                
                st.markdown(f"[**View Live Diagram**]({live_editor_url})")
                
            except Exception:
                pass  # If encoding fails, just skip the live link
        
        with col2:
            st.markdown("**Quick Links:**")
            st.markdown("[Mermaid Live Editor](https://mermaid.live)")
            st.markdown("[Mermaid Documentation](https://mermaid.js.org/)")
        
        # Create alternative text-based visualization
        _create_text_based_diagram(diagram_code, title)

def _create_text_based_diagram(diagram_code: str, title: str):
    """Create a simple text-based representation of the diagram"""
    try:
        st.markdown("### Text-Based Visualization")
        
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
                    nodes.append(f"Package: {node_text}")
        
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
    page_title="Code Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force Dark Theme and Modern Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Reset and Force Dark Theme */
    * {
        color: #ffffff !important;
    }
    
    /* Force dark background on main app */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0066ff 100%) !important;
        background-attachment: fixed !important;
        min-height: 100vh !important;
        position: relative !important;
    }
    
    /* Animated background overlay */
    .stApp::before {
        content: '' !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        background: 
            repeating-linear-gradient(
                45deg,
                transparent,
                transparent 2px,
                rgba(0, 102, 255, 0.05) 2px,
                rgba(0, 102, 255, 0.05) 4px
            ) !important;
        z-index: -1 !important;
        animation: backgroundShift 20s ease-in-out infinite !important;
        pointer-events: none !important;
    }
    
    /* Floating particles */
    .stApp::after {
        content: '' !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(0, 255, 136, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(0, 102, 255, 0.1) 0%, transparent 50%) !important;
        z-index: 0 !important;
        animation: particleFloat 15s ease-in-out infinite !important;
        pointer-events: none !important;
    }
    
    @keyframes backgroundShift {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    @keyframes particleFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-10px) rotate(1deg); }
        66% { transform: translateY(5px) rotate(-0.5deg); }
    }
    
    /* Main container styling */
    .main .block-container {
        background: rgba(15, 15, 35, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(0, 102, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        margin-top: 1rem !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* Header animations */
    .main-title {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #00ff88, #0066ff, #8b5cf6) !important;
        background-size: 300% 300% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        animation: gradientShift 4s ease-in-out infinite !important;
        text-align: center !important;
        margin: 2rem 0 !important;
        position: relative !important;
    }
    
    .main-title::before {
        content: '< ' !important;
        color: #00ff88 !important;
        animation: pulse 2s ease-in-out infinite !important;
        -webkit-text-fill-color: #00ff88 !important;
    }
    
    .main-title::after {
        content: ' />' !important;
        color: #8b5cf6 !important;
        animation: pulse 2s ease-in-out infinite reverse !important;
        -webkit-text-fill-color: #8b5cf6 !important;
    }
    
    .subtitle {
        font-size: 1.3rem !important;
        color: rgba(255, 255, 255, 0.8) !important;
        text-align: center !important;
        margin-bottom: 3rem !important;
        animation: fadeInUp 1s ease-out 0.5s both !important;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced Buttons with stronger selectors */
    .stButton > button,
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0066ff, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(0, 102, 255, 0.3) !important;
    }
    
    .stButton > button:hover,
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(0, 102, 255, 0.4) !important;
        background: linear-gradient(135deg, #0052cc, #7c3aed) !important;
    }
    
    /* Input Styling with glassmorphism */
    .stTextInput > div > div > input,
    div[data-testid="textInput"] input {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    div[data-testid="textInput"] input:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 0 2px rgba(0, 255, 136, 0.3) !important;
        transform: scale(1.02) !important;
        background: rgba(255, 255, 255, 0.15) !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 16px !important;
        padding: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        padding: 12px 24px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #00ff88 !important;
        background: rgba(0, 255, 136, 0.15) !important;
        transform: translateY(-1px) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ff88, #0066ff) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 15px 35px rgba(0, 255, 136, 0.15) !important;
        border-color: rgba(0, 255, 136, 0.4) !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        margin-top: 0.5rem !important;
    }
    
    /* Success/Error Messages */
    .stAlert > div {
        background: rgba(0, 255, 136, 0.15) !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
        color: #00ff88 !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        animation: fadeInUp 0.5s ease-out !important;
    }
    
    /* Override any remaining light theme elements */
    .stMarkdown, .stText, .stJson, .stDataFrame {
        color: #ffffff !important;
        background: transparent !important;
    }
    
    /* Sidebar removal - hide completely */
    .css-1d391kg, .css-1lcbmhc, .css-17ziqus, .css-1y4p8pa {
        display: none !important;
    }
    
    .quality-score-b {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2em;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
    }
    
    .quality-score-c {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2em;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.3);
    }
    
    .quality-score-d {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2em;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(249, 115, 22, 0.3);
    }
    
    .quality-score-f {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2em;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
    }
    
    .recommendation-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        color: #ffffff;
    }
    
    .stSelectbox > div > div {
        background: rgba(45, 55, 72, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input {
        background: rgba(45, 55, 72, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #ffffff;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: rgba(30, 39, 46, 0.9);
        backdrop-filter: blur(10px);
    }
    
    .stExpander {
        background: rgba(45, 55, 72, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stMetric {
        background: rgba(45, 55, 72, 0.6);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .stMarkdown {
        color: #cbd5e0;
    }
</style>
""", unsafe_allow_html=True)

# Creative Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">CodeScope</h1>
    <p class="subtitle">AI-Powered Repository Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

class AnalysisCache:
    """Smart caching system for analysis results"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, repo_url: str, analysis_type: str = "full") -> str:
        """Generate cache key for repository"""
        key_string = f"{repo_url}_{analysis_type}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get_cached_analysis(self, repo_url: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Get cached analysis if available and fresh"""
        cache_key = self._get_cache_key(repo_url)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still fresh
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                return None
            
            return cached_data['analysis']
        except Exception:
            return None
    
    def save_analysis(self, repo_url: str, analysis_result: Dict):
        """Save analysis result to cache"""
        cache_key = self._get_cache_key(repo_url)
        cache_path = self._get_cache_path(cache_key)
        
        cached_data = {
            'timestamp': datetime.now().isoformat(),
            'repo_url': repo_url,
            'analysis': analysis_result
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cached_data, f, indent=2)
        except Exception:
            pass  # Fail silently if caching fails

class RepositoryAnalyzer:
    """Enhanced repository analyzer with hybrid Gemini/Ollama approach"""
    
    def __init__(self, gemini_api_keys: List[str] = None):
        # Initialize model services
        from model_service import GeminiService, OllamaService
        
        self.gemini_service = None
        self.api_requests_made = 0
        
        # Initialize Gemini service if API keys are provided
        if gemini_api_keys and GEMINI_AVAILABLE:
            try:
                self.gemini_service = GeminiService(gemini_api_keys)
                if self.gemini_service.is_available():
                    # Gemini service initialized silently
                    pass
                else:
                    # Gemini service initialized but no keys are available - silent
                    pass
            except Exception as e:
                st.error(f"Failed to initialize Gemini service: {e}")
                self.gemini_service = None
        
        # Initialize Ollama
        self.ollama_available = self._check_ollama()
        
        # Initialize analyzers
        self.quality_analyzer = CodeQualityAnalyzer()
        
        # Initialize cache system
        self.cache = AnalysisCache()
    
    @property
    def model_service(self):
        """Get the available model service (Gemini preferred, Ollama fallback)"""
        if self.gemini_service and self.gemini_service.is_available():
            return self.gemini_service
        # Fallback to Ollama if available
        if self.ollama_service and self.ollama_service.is_available():
            return self.ollama_service
        # Return None if no service is available
        return None
    
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
        if not self.gemini_service or not self.gemini_service.is_available():
            return None
        
        try:
            response = await self.gemini_service.generate_response(messages, max_retries=max_retries)
            if response:
                self.api_requests_made += 1
                return response
            else:
                return None
        except Exception as e:
            # Check if all keys are exhausted
            if "All Gemini API keys" in str(e):
                st.warning("Warning: All Gemini API keys are rate limited, switching to Ollama")
            return None
    
    async def _call_ollama(self, messages: List[Dict], model: str = None) -> str:
        """Call Ollama API"""
        import subprocess
        import json
        
        # Use selected model from session state if not specified
        if model is None:
            model = st.session_state.get('selected_ollama_model', 'llama3.2:3b')
        
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
        if self.gemini_service and self.gemini_service.is_available():
            try:
                response = await self._call_gemini(messages)
                if response:
                    return response
                else:
                    # If Gemini fails, log it and fallback to Ollama
                    if self.api_limit_reached:
                        st.info("API quota reached, switching to local Ollama...")
                    else:
                        st.info("Gemini API failed, switching to local Ollama...")
            except Exception as e:
                st.info(f"Gemini error: {str(e)}, switching to local Ollama...")
        
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
        """Clone repository with shallow cloning for faster performance"""
        temp_dir = tempfile.mkdtemp()
        repo_url = f"https://github.com/{repo_info['owner']}/{repo_info['repo']}.git"
        
        try:
            # Shallow clone with depth=1 (only latest commit)
            git.Repo.clone_from(
                repo_url, 
                temp_dir, 
                depth=1,  # Only latest commit
                single_branch=True,  # Only default branch
                no_checkout=False
            )
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
        """Analyze repository structure with parallel processing for faster performance"""
        # First, quickly scan for all files
        all_files = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
            for file in files:
                if not file.startswith('.'):
                    all_files.append(os.path.join(root, file))
        
        # Process files in parallel batches
        batch_size = 10  # Process 10 files at once
        detailed_files = {}
        file_types = {}
        total_files = 0
        total_lines = 0
        key_files = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(0, len(all_files), batch_size):
                batch = all_files[i:i + batch_size]
                
                # Create partial function with repo_path
                process_func = partial(self._process_single_file, repo_path)
                
                # Submit all files in batch
                future_to_file = {executor.submit(process_func, file_path): file_path for file_path in batch}
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            relative_path = os.path.relpath(file_path, repo_path)
                            detailed_files[relative_path] = result
                            
                            # Update counters
                            file_ext = result.get('extension', '')
                            file_types[file_ext] = file_types.get(file_ext, 0) + 1
                            total_files += 1
                            total_lines += result.get('lines', 0)
                            
                            # Check for key files
                            file_name = Path(file_path).name.lower()
                            if file_name in ['readme.md', 'requirements.txt', 'package.json', 'dockerfile', 'setup.py']:
                                key_files.append(relative_path)
                    except Exception:
                        continue  # Skip problematic files
        
        # Store context for enhanced analysis
        self._current_detailed_files = detailed_files
        self._current_technologies = self._detect_technologies(file_types, key_files)
        self._total_files_count = total_files
        self._key_files = key_files
        
        # Build analysis structure
        analysis = {
            "file_structure": {
                "total_files": total_files,
                "file_types": file_types,
                "total_lines": total_lines
            },
            "technologies": self._current_technologies,
            "key_files": key_files,
            "detailed_files": detailed_files,
            "quality_metrics": {}
        }
        
        # Analyze code quality metrics
        analysis["quality_metrics"] = self._analyze_quality_metrics(detailed_files)
        
        return analysis
    
    def _process_single_file(self, repo_path: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single file - designed to be thread-safe"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Skip non-code files early
            if file_ext not in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php', '.cs', '.jsx', '.tsx', '.md', '.txt', '.json', '.yaml', '.yml']:
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = len(content.splitlines())
                
                if lines == 0:
                    return None
                
                # Analyze code files
                if file_ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php', '.cs', '.jsx', '.tsx']:
                    return {
                        "extension": file_ext,
                        "lines": lines,
                        "content_preview": content[:2000],
                        "full_content": content,
                        "imports": self._extract_imports(content, file_ext),
                        "functions": self._extract_functions(content, file_ext),
                        "classes": self._extract_classes(content, file_ext)
                    }
                
                # Analyze important non-code files
                elif file_ext in ['.md', '.txt', '.json', '.yaml', '.yml'] or Path(file_path).name in ['README', 'LICENSE']:
                    if len(content.strip()) > 0:
                        return {
                            "extension": file_ext,
                            "lines": lines,
                            "content_preview": content[:1000],
                            "file_type": "configuration" if file_ext in ['.json', '.yaml', '.yml'] else "documentation"
                        }
                
                return None
        except Exception:
            return None
    
    def _detect_technologies(self, file_types: Dict[str, int], key_files: List[str]) -> List[str]:
        """Detect technologies used in the repository"""
        technologies = []
        
        if '.py' in file_types: technologies.append("Python")
        if '.js' in file_types or '.jsx' in file_types: technologies.append("JavaScript")
        if '.ts' in file_types or '.tsx' in file_types: technologies.append("TypeScript")
        if '.java' in file_types: technologies.append("Java")
        if '.cpp' in file_types or '.c' in file_types: technologies.append("C/C++")
        if '.go' in file_types: technologies.append("Go")
        if '.rs' in file_types: technologies.append("Rust")
        if '.rb' in file_types: technologies.append("Ruby")
        if '.php' in file_types: technologies.append("PHP")
        if '.cs' in file_types: technologies.append("C#")
        
        # Check for specific frameworks/tools
        key_files_lower = [f.lower() for f in key_files]
        if 'package.json' in key_files_lower: technologies.append("Node.js")
        if 'requirements.txt' in key_files_lower: technologies.append("Python Package")
        if 'dockerfile' in key_files_lower: technologies.append("Docker")
        if 'setup.py' in key_files_lower: technologies.append("Python Package")
        
        return technologies
    
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
        
        
        # Generate overall quality score (now includes all metrics)
        quality_metrics["overall_score"] = self.quality_analyzer.generate_quality_score(quality_metrics)
        
        return quality_metrics
    
    async def _explain_single_file(self, file_path: str, file_info: Dict[str, Any]) -> str:
        """Generate explanation for a single file with comprehensive repository context"""
        if (not self.gemini_service or not self.gemini_service.is_available()) and not self.ollama_available:
            return f"File analysis requires either Gemini API or Ollama to be running. File has {file_info.get('lines', 0)} lines."
        
        # Build comprehensive context including the entire repository understanding
        context = f"""
REPOSITORY CONTEXT:
===================
Target File: {file_path}
File Extension: {file_info.get('extension', 'unknown')}
File Size: {file_info.get('lines', 0)} lines

CURRENT FILE CONTENT:
=====================
{file_info.get('content_preview', '')[:2000]}

CURRENT FILE STRUCTURE:
=======================
Imports: {', '.join(file_info.get('imports', [])[:10])}
Functions: {', '.join(file_info.get('functions', [])[:10])}
Classes: {', '.join(file_info.get('classes', [])[:8])}

REPOSITORY ARCHITECTURE CONTEXT:
=================================
Technologies Detected: {', '.join(getattr(self, '_current_technologies', []))}
Total Repository Files: {getattr(self, '_total_files_count', 'Unknown')}
Key Repository Files: {', '.join(getattr(self, '_key_files', [])[:5])}

PROJECT TYPE INDICATORS:
========================
- Has ML/AI patterns: {self._detect_ml_patterns(file_info)}
- Has web framework patterns: {self._detect_web_patterns(file_info)}  
- Has data processing patterns: {self._detect_data_patterns(file_info)}
- Has CLI application patterns: {self._detect_cli_patterns(file_info)}

CROSS-FILE RELATIONSHIPS:
=========================
Files that import this: {self._find_files_importing(file_path)}
Files this imports from: {', '.join(file_info.get('imports', [])[:8])}
Similar files in project: {self._find_similar_files(file_path, file_info)}

ARCHITECTURAL PATTERNS DETECTED:
================================
{self._detect_architectural_patterns_for_file(file_path, file_info)}
"""
        
        prompt = f"""
You are an expert code analyst with deep understanding of software architecture and design patterns. 
Analyze this file within its complete repository context.

{context}

Provide a comprehensive explanation (5-7 detailed sentences, 300-500 words) covering:

1. **Purpose & Role**: What this file does and its specific role in the project architecture. Consider the project type (ML, web app, CLI, etc.) and explain how this file fits into that domain.

2. **Key Components**: For each major function/class (up to 5), explain:
   - What it does and why it exists
   - Input parameters and expected outputs  
   - How it works internally (algorithm/logic)
   - Edge cases it handles or should handle
   - How it relates to other components

3. **Architectural Relationships**: 
   - How this file interacts with other parts of the system
   - What design patterns are used (MVC, Factory, Observer, etc.)
   - Dependencies and what depends on this file

4. **Code Quality & Design**:
   - Well-implemented aspects and best practices used
   - Potential improvements or concerns
   - How it follows or violates project conventions

5. **Domain-Specific Insights**: 
   - If ML: explain the model/data flow aspects
   - If web: explain request/response handling
   - If CLI: explain command processing
   - If data: explain processing pipeline role

Use specific examples from the code when helpful. Focus on understanding the INTENT and CONTEXT, not just describing syntax.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a senior software architect and code analyst who understands both code structure and business context."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._call_llm_hybrid(messages, "file analysis")
            return response.strip()
            
        except Exception as e:
            return f"File analysis failed: {str(e)}"
    
    def _detect_ml_patterns(self, file_info: Dict[str, Any]) -> bool:
        """Detect if file contains ML/AI patterns"""
        content = file_info.get('content_preview', '').lower()
        imports = ' '.join(file_info.get('imports', [])).lower()
        functions = ' '.join(file_info.get('functions', [])).lower()
        
        ml_indicators = ['torch', 'tensorflow', 'sklearn', 'model', 'train', 'predict', 'neural', 'gradient', 'loss', 'optimizer', 'dataset', 'embedding']
        return any(indicator in content + imports + functions for indicator in ml_indicators)
    
    def _detect_web_patterns(self, file_info: Dict[str, Any]) -> bool:
        """Detect if file contains web framework patterns"""
        content = file_info.get('content_preview', '').lower()
        imports = ' '.join(file_info.get('imports', [])).lower()
        
        web_indicators = ['flask', 'django', 'fastapi', 'streamlit', 'request', 'response', 'route', 'endpoint', 'api', 'http']
        return any(indicator in content + imports for indicator in web_indicators)
    
    def _detect_data_patterns(self, file_info: Dict[str, Any]) -> bool:
        """Detect if file contains data processing patterns"""
        content = file_info.get('content_preview', '').lower()
        imports = ' '.join(file_info.get('imports', [])).lower()
        functions = ' '.join(file_info.get('functions', [])).lower()
        
        data_indicators = ['pandas', 'numpy', 'csv', 'json', 'dataframe', 'process', 'transform', 'parse', 'load', 'save']
        return any(indicator in content + imports + functions for indicator in data_indicators)
    
    def _detect_cli_patterns(self, file_info: Dict[str, Any]) -> bool:
        """Detect if file contains CLI application patterns"""
        content = file_info.get('content_preview', '').lower()
        imports = ' '.join(file_info.get('imports', [])).lower()
        
        cli_indicators = ['argparse', 'click', 'main', 'argv', 'command', 'parser', 'args']
        return any(indicator in content + imports for indicator in cli_indicators)
    
    def _find_files_importing(self, target_file: str) -> str:
        """Find other files that import the target file"""
        if not hasattr(self, '_current_detailed_files'):
            return "Unknown"
        
        importing_files = []
        target_module = Path(target_file).stem
        
        for file_path, file_info in self._current_detailed_files.items():
            imports = file_info.get('imports', [])
            if any(target_module in imp for imp in imports):
                importing_files.append(Path(file_path).name)
        
        return ', '.join(importing_files[:5]) if importing_files else "None detected"
    
    def _find_similar_files(self, target_file: str, file_info: Dict[str, Any]) -> str:
        """Find files with similar patterns or purposes"""
        if not hasattr(self, '_current_detailed_files'):
            return "Unknown"
        
        similar_files = []
        target_functions = set(file_info.get('functions', []))
        target_imports = set(file_info.get('imports', []))
        
        for file_path, other_info in self._current_detailed_files.items():
            if file_path == target_file:
                continue
                
            other_functions = set(other_info.get('functions', []))
            other_imports = set(other_info.get('imports', []))
            
            # Calculate similarity based on shared imports and function patterns
            import_similarity = len(target_imports & other_imports) / max(len(target_imports), 1)
            function_similarity = len(target_functions & other_functions) / max(len(target_functions), 1)
            
            if import_similarity > 0.3 or function_similarity > 0.2:
                similar_files.append(Path(file_path).name)
        
        return ', '.join(similar_files[:4]) if similar_files else "None detected"
    
    def _detect_architectural_patterns_for_file(self, file_path: str, file_info: Dict[str, Any]) -> str:
        """Detect architectural patterns relevant to this specific file"""
        patterns = []
        
        file_name = Path(file_path).name.lower()
        content = file_info.get('content_preview', '').lower()
        functions = file_info.get('functions', [])
        classes = file_info.get('classes', [])
        
        # Pattern detection based on file characteristics
        if 'model' in file_name or any('model' in cls.lower() for cls in classes):
            patterns.append("Data Model / Domain Model pattern")
        
        if 'controller' in file_name or 'handler' in file_name:
            patterns.append("Controller / Handler pattern")
        
        if 'view' in file_name or 'template' in file_name:
            patterns.append("View / Presentation pattern")
        
        if 'service' in file_name or any('service' in func.lower() for func in functions):
            patterns.append("Service Layer pattern")
        
        if 'factory' in content or any('create' in func.lower() for func in functions):
            patterns.append("Factory / Builder pattern")
        
        if 'config' in file_name or 'settings' in file_name:
            patterns.append("Configuration pattern")
        
        if 'util' in file_name or 'helper' in file_name:
            patterns.append("Utility / Helper pattern")
        
        return '; '.join(patterns) if patterns else "No specific patterns detected"
    
    def _prioritize_files(self, detailed_files: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Prioritize files for analysis based on importance"""
        
        priority_scores = {}
        
        for file_path, file_info in detailed_files.items():
            score = 0
            file_name = Path(file_path).name.lower()
            
            # High priority files
            if file_name in ['main.py', 'app.py', 'index.js', 'server.py', 'manage.py']:
                score += 100
            elif file_name in ['readme.md', 'package.json', 'requirements.txt', 'setup.py']:
                score += 90
            elif file_name.startswith('test_') or '/test' in file_path:
                score += 20  # Lower priority for tests
            
            # Score by file size (bigger files are often more important)
            lines = file_info.get('lines', 0)
            if lines > 100:
                score += 30
            elif lines > 50:
                score += 20
            elif lines > 20:
                score += 10
            
            # Score by file type
            ext = file_info.get('extension', '')
            if ext in ['.py', '.js', '.ts', '.java']:
                score += 25
            elif ext in ['.md', '.txt']:
                score += 15
            
            # Boost for files with many functions/classes
            functions = file_info.get('functions', [])
            classes = file_info.get('classes', [])
            if len(functions) + len(classes) > 5:
                score += 20
            elif len(functions) + len(classes) > 2:
                score += 10
            
            priority_scores[file_path] = score
        
        # Sort by priority score
        return sorted(detailed_files.items(), key=lambda x: priority_scores.get(x[0], 0), reverse=True)
    
    async def _generate_file_explanations(self, detailed_files: Dict[str, Any]) -> Dict[str, str]:
        """Generate explanations with intelligent prioritization"""
        if (not self.gemini_service or not self.gemini_service.is_available()) and not self.ollama_available:
            return {}
        
        # Prioritize files
        prioritized_files = self._prioritize_files(detailed_files)
        
        # Start with top 12 most important files
        important_files = prioritized_files[:12]
        
        file_explanations = {}
        
        # Process important files first
        for file_path, file_info in important_files:
            try:
                explanation = await self._explain_single_file(file_path, file_info)
                if explanation:
                    file_explanations[file_path] = explanation
            except Exception as e:
                file_explanations[file_path] = f"Analysis failed: {str(e)}"
        
        return file_explanations
    
    async def _explain_files_batch(self, file_batch: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, str]:
        """Explain multiple files in a single AI request"""
        
        if not file_batch:
            return {}
        
        # Build batch context
        batch_context = "Analyze these files from a code repository:\n\n"
        
        for i, (file_path, file_info) in enumerate(file_batch, 1):
            batch_context += f"FILE {i}: {file_path}\n"
            batch_context += f"Lines: {file_info.get('lines', 0)}\n"
            batch_context += f"Content preview:\n{file_info.get('content_preview', '')[:800]}\n"
            batch_context += "---\n\n"
        
        prompt = f"""
        Analyze each file above and provide a detailed explanation for each one.
        
        {batch_context}
        
        For each file, provide:
        1. Purpose and role in the project
        2. Key functions/classes and what they do
        3. How it interacts with other components
        4. Any notable patterns or concerns
        
        Format your response as:
        
        FILE 1: [filename]
        [Your detailed explanation here]
        
        FILE 2: [filename]
        [Your detailed explanation here]
        
        And so on...
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a code analyst providing detailed file explanations."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._call_llm_hybrid(messages, "batch file analysis")
            
            # Parse batch response back to individual file explanations
            return self._parse_batch_response(response, file_batch)
            
        except Exception as e:
            # Fallback to individual explanations for this batch
            explanations = {}
            for file_path, file_info in file_batch:
                explanations[file_path] = f"Batch analysis failed, file has {file_info.get('lines', 0)} lines"
            return explanations

    def _parse_batch_response(self, response: str, file_batch: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, str]:
        """Parse batch AI response back to individual file explanations"""
        
        explanations = {}
        
        # Split response by FILE markers
        sections = response.split('FILE ')
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            if i <= len(file_batch):
                file_path = file_batch[i-1][0]
                
                # Extract explanation (everything after the first line)
                lines = section.split('\n')
                if len(lines) > 1:
                    explanation = '\n'.join(lines[1:]).strip()
                    explanations[file_path] = explanation
                else:
                    explanations[file_path] = f"Brief analysis: {section.strip()}"
        
        # Fill in any missing explanations
        for file_path, file_info in file_batch:
            if file_path not in explanations:
                explanations[file_path] = f"Code file with {file_info.get('lines', 0)} lines"
        
        return explanations
    
    async def _generate_overall_insights(self, analysis: Dict[str, Any]) -> str:
        """Generate overall insights about the repository with comprehensive contextual analysis"""
        if (not self.gemini_service or not self.gemini_service.is_available()) and not self.ollama_available:
            return "Overall analysis requires either Gemini API or Ollama to be running."
        
        file_structure = analysis['file_structure']
        quality_metrics = analysis.get('quality_metrics', {})
        detailed_files = analysis.get('detailed_files', {})
        
        # Build comprehensive repository intelligence
        repo_intelligence = self._build_repository_intelligence(detailed_files, analysis)
        
        # Build comprehensive context
        context = f"""
REPOSITORY COMPREHENSIVE ANALYSIS:
==================================

QUANTITATIVE METRICS:
----------------------
- Total files: {file_structure['total_files']}
- Total lines of code: {file_structure['total_lines']}
- File type distribution: {', '.join([f"{ext}({count})" for ext, count in file_structure['file_types'].items()])}
- Technologies detected: {', '.join(analysis['technologies']) if analysis['technologies'] else 'None detected'}
- Key entry point files: {', '.join(analysis['key_files']) if analysis['key_files'] else 'None detected'}

CODE QUALITY INTELLIGENCE:
---------------------------
- Average complexity: {quality_metrics.get('complexity', {}).get('average_complexity', 'N/A')}
- Code duplication: {quality_metrics.get('duplication', {}).get('duplication_percentage', 'N/A')}% 
- Comment ratio: {quality_metrics.get('comment_ratio', 'N/A')}%
- Quality score: {quality_metrics.get('overall_score', {}).get('score', 'N/A')}/100 (Grade: {quality_metrics.get('overall_score', {}).get('grade', 'N/A')})

ARCHITECTURAL INTELLIGENCE:
----------------------------
Project Type Indicators:
{repo_intelligence['project_type_analysis']}

Detected Architecture Patterns:
{repo_intelligence['architecture_patterns']}

Component Interaction Patterns:
{repo_intelligence['interaction_patterns']}

CODEBASE INTELLIGENCE:
----------------------
Domain-Specific Patterns:
{repo_intelligence['domain_patterns']}

File Purpose Distribution:
{repo_intelligence['file_purposes']}

Dependency Relationship Analysis:
{repo_intelligence['dependency_analysis']}

COMPLEXITY & MAINTAINABILITY INTELLIGENCE:
-------------------------------------------
{repo_intelligence['complexity_insights']}

TECHNOLOGY STACK INTELLIGENCE:
-------------------------------
{repo_intelligence['tech_stack_analysis']}
"""
        
        prompt = f"""
You are a senior software architect and technical lead conducting a comprehensive repository assessment. 
Based on the detailed analysis below, provide strategic insights about this codebase.

{context}

Provide a comprehensive assessment in MARKDOWN format with the following structure:

## **Project Identity & Purpose**
Analyze what type of project this is based on the patterns, technologies, and file structures detected. Don't guess - use the actual evidence from the codebase. Explain the project's scope, primary goals, and target domain.

## **Architectural Assessment**
Evaluate the architectural approach based on detected patterns, file organization, and component relationships. Comment on:
- How well the architecture fits the project's purpose
- Strengths in the current design approach
- Any architectural debt or inconsistencies observed

## **Code Quality & Maintainability**
Analyze the quality metrics in context:
- What the quality scores reveal about development practices
- How complexity/duplication/comments affect maintainability  
- Evidence of good or problematic coding practices
- How the quality supports or hinders the project goals

## **Strategic Recommendations**
Provide 3-5 specific, actionable recommendations that would have the highest impact:
- Technical debt to address
- Architectural improvements to consider
- Quality practices to implement
- Strategic enhancements for the project type

Focus on insights that require understanding the ENTIRE codebase context, not just individual files. 
Reference specific patterns, files, and metrics from the analysis. Aim for 400-600 words of strategic value.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a senior software architect providing strategic codebase assessment based on comprehensive analysis."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._call_llm_hybrid(messages, "overall analysis")
            return response.strip()
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def _build_repository_intelligence(self, detailed_files: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive repository intelligence by analyzing patterns across all files"""
        intelligence = {
            'project_type_analysis': '',
            'architecture_patterns': '',
            'interaction_patterns': '',
            'domain_patterns': '',
            'file_purposes': '',
            'dependency_analysis': '',
            'complexity_insights': '',
            'tech_stack_analysis': ''
        }
        
        # Analyze project type indicators
        ml_files = sum(1 for f in detailed_files.values() if self._detect_ml_patterns(f))
        web_files = sum(1 for f in detailed_files.values() if self._detect_web_patterns(f))
        data_files = sum(1 for f in detailed_files.values() if self._detect_data_patterns(f))
        cli_files = sum(1 for f in detailed_files.values() if self._detect_cli_patterns(f))
        
        total_files = len(detailed_files)
        
        project_indicators = []
        if ml_files > 0:
            project_indicators.append(f"Machine Learning ({ml_files}/{total_files} files with ML patterns)")
        if web_files > 0:
            project_indicators.append(f"Web Application ({web_files}/{total_files} files with web patterns)")
        if data_files > 0:
            project_indicators.append(f"Data Processing ({data_files}/{total_files} files with data patterns)")
        if cli_files > 0:
            project_indicators.append(f"CLI Application ({cli_files}/{total_files} files with CLI patterns)")
        
        intelligence['project_type_analysis'] = '; '.join(project_indicators) if project_indicators else "General Purpose Application"
        
        # Analyze architecture patterns
        architecture_patterns = []
        file_names = [Path(f).name.lower() for f in detailed_files.keys()]
        
        if any('model' in name for name in file_names):
            architecture_patterns.append("Model Layer Pattern")
        if any('view' in name or 'template' in name for name in file_names):
            architecture_patterns.append("View Layer Pattern")
        if any('controller' in name or 'handler' in name for name in file_names):
            architecture_patterns.append("Controller/Handler Pattern")
        if any('service' in name for name in file_names):
            architecture_patterns.append("Service Layer Pattern")
        if any('config' in name or 'settings' in name for name in file_names):
            architecture_patterns.append("Configuration Pattern")
        
        intelligence['architecture_patterns'] = '; '.join(architecture_patterns) if architecture_patterns else "No clear architectural patterns detected"
        
        # Analyze interaction patterns
        import_counts = {}
        for file_info in detailed_files.values():
            for imp in file_info.get('imports', []):
                import_counts[imp] = import_counts.get(imp, 0) + 1
        
        common_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        interaction_patterns = [f"{imp} (used in {count} files)" for imp, count in common_imports]
        intelligence['interaction_patterns'] = '; '.join(interaction_patterns) if interaction_patterns else "Limited cross-file dependencies"
        
        # Analyze domain patterns
        domain_patterns = []
        all_functions = []
        for file_info in detailed_files.values():
            all_functions.extend(file_info.get('functions', []))
        
        function_text = ' '.join(all_functions).lower()
        
        if 'train' in function_text or 'model' in function_text:
            domain_patterns.append("Machine Learning Training/Inference")
        if 'api' in function_text or 'endpoint' in function_text:
            domain_patterns.append("API/Web Service")
        if 'process' in function_text or 'transform' in function_text:
            domain_patterns.append("Data Processing Pipeline")
        if 'parse' in function_text or 'load' in function_text:
            domain_patterns.append("Data Input/Output")
        
        intelligence['domain_patterns'] = '; '.join(domain_patterns) if domain_patterns else "General utility functions"
        
        # Analyze file purposes
        purpose_distribution = {}
        for file_path in detailed_files.keys():
            file_name = Path(file_path).name.lower()
            if 'main' in file_name or 'app' in file_name:
                purpose_distribution['Entry Points'] = purpose_distribution.get('Entry Points', 0) + 1
            elif 'test' in file_name:
                purpose_distribution['Tests'] = purpose_distribution.get('Tests', 0) + 1
            elif 'config' in file_name or 'settings' in file_name:
                purpose_distribution['Configuration'] = purpose_distribution.get('Configuration', 0) + 1
            elif 'util' in file_name or 'helper' in file_name:
                purpose_distribution['Utilities'] = purpose_distribution.get('Utilities', 0) + 1
            else:
                purpose_distribution['Core Logic'] = purpose_distribution.get('Core Logic', 0) + 1
        
        intelligence['file_purposes'] = '; '.join([f"{purpose}: {count}" for purpose, count in purpose_distribution.items()])
        
        # Analyze dependencies
        total_imports = sum(len(f.get('imports', [])) for f in detailed_files.values())
        avg_imports = total_imports / len(detailed_files) if detailed_files else 0
        intelligence['dependency_analysis'] = f"Average {avg_imports:.1f} imports per file, indicating {'high' if avg_imports > 5 else 'moderate' if avg_imports > 2 else 'low'} coupling"
        
        # Complexity insights
        total_functions = sum(len(f.get('functions', [])) for f in detailed_files.values())
        avg_functions = total_functions / len(detailed_files) if detailed_files else 0
        intelligence['complexity_insights'] = f"Average {avg_functions:.1f} functions per file, suggesting {'complex' if avg_functions > 10 else 'moderate' if avg_functions > 5 else 'simple'} file organization"
        
        # Tech stack analysis
        technologies = analysis.get('technologies', [])
        if technologies:
            primary_tech = technologies[0] if technologies else "Unknown"
            intelligence['tech_stack_analysis'] = f"Primary: {primary_tech}, Supporting: {', '.join(technologies[1:4])}"
        else:
            intelligence['tech_stack_analysis'] = "No specific technology stack detected"
        
        return intelligence
    
    async def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """Analyze a GitHub repository with enhanced quality metrics and proper error handling"""
        repo_path = None  # Initialize variable for cleanup
        # Check cache first
        cached_result = self.cache.get_cached_analysis(repo_url)
        if cached_result:
            return cached_result

        try:
            # Parse repository info
            repo_info = self._parse_repo_url(repo_url)
            if not repo_info:
                return {"error": "Invalid repository URL format", "success": False}

            # Clone repository
            repo_path = self._clone_repository(repo_info)

            # Analyze code structure in thread pool
            analysis = await asyncio.get_event_loop().run_in_executor(
                None, self._analyze_code_structure, repo_path
            )

            # Generate file explanations
            file_explanations = await self._generate_file_explanations(analysis.get("detailed_files", {}))
            analysis["file_explanations"] = file_explanations


            # Generate overall insights
            insights = await self._generate_overall_insights(analysis)

            result = {
                "repository": f"{repo_info['owner']}/{repo_info['repo']}",
                "analysis": analysis,
                "insights": insights,
                "success": True
            }

            # Cache the result for future requests
            self.cache.save_analysis(repo_url, result)
            return result

        except Exception as e:
            return {"error": str(e), "success": False}

        finally:
            # Cleanup cloned repository safely
            if repo_path and os.path.exists(repo_path):
                try:
                    shutil.rmtree(repo_path, ignore_errors=True)
                except Exception as cleanup_error:
                    st.warning(f"Cleanup warning: {str(cleanup_error)}")
    
    def analyze_repository_progressive(self, repo_url: str):
        """Analyze repository with progressive loading and real-time updates"""
        
        # Create placeholders for different sections
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Section placeholders
        basic_info_placeholder = st.empty()
        file_structure_placeholder = st.empty()
        file_explanations_placeholder = st.empty()
        insights_placeholder = st.empty()
        
        try:
            # Check cache first
            status_text.text("Checking cache...")
            progress_bar.progress(5)
            
            cached_result = self.cache.get_cached_analysis(repo_url)
            if cached_result:
                status_text.text("Found cached analysis!")
                progress_bar.progress(100)
                return cached_result
            
            # Step 1: Parse URL and Clone repository (20% progress)
            status_text.text("Cloning repository...")
            progress_bar.progress(20)
            
            repo_info = self._parse_repo_url(repo_url)
            if not repo_info:
                st.error("Invalid repository URL format")
                return
            
            repo_path = self._clone_repository(repo_info)
            
            # Step 2: Basic analysis (40% progress)
            status_text.text("Analyzing file structure...")
            progress_bar.progress(40)
            
            analysis = self._analyze_code_structure(repo_path)
            
            # Show basic info immediately
            with basic_info_placeholder.container():
                st.subheader("Repository Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", analysis["file_structure"]["total_files"])
                with col2:
                    st.metric("Lines of Code", analysis["file_structure"]["total_lines"])
                with col3:
                    st.metric("Technologies", len(analysis["technologies"]))
                
                if analysis["technologies"]:
                    st.write("**Technologies detected:**", ", ".join(analysis["technologies"]))
            
            # Step 3: File structure display (50% progress)
            status_text.text("Displaying file structure...")
            progress_bar.progress(50)
            
            with file_structure_placeholder.container():
                st.subheader("File Structure")
                if analysis["file_structure"]["file_types"]:
                    # Create a simple visualization of file types
                    file_types_df = pd.DataFrame(
                        list(analysis["file_structure"]["file_types"].items()),
                        columns=['Extension', 'Count']
                    )
                    st.bar_chart(file_types_df.set_index('Extension'))
            
            # Step 4: File explanations (70% progress)
            status_text.text("Generating file explanations...")
            progress_bar.progress(70)
            
            file_explanations = asyncio.run(self._generate_file_explanations(analysis["detailed_files"]))
            analysis["file_explanations"] = file_explanations
            
            # Show file explanations immediately
            with file_explanations_placeholder.container():
                st.subheader("Key File Explanations")
                for file_path, explanation in file_explanations.items():
                    with st.expander(f"File: {file_path}"):
                        st.write(explanation)
            
            # Step 5: Overall insights (90% progress)
            status_text.text("Generating insights...")
            progress_bar.progress(90)
            
            insights = asyncio.run(self._generate_overall_insights(analysis))
            
            # Show insights immediately
            with insights_placeholder.container():
                st.subheader("AI Insights")
                st.write(insights)
            
            # Step 6: Save to cache and complete (100% progress)
            result = {
                "repository": f"{repo_info['owner']}/{repo_info['repo']}",
                "analysis": analysis,
                "insights": insights,
                "success": True
            }
            
            self.cache.save_analysis(repo_url, result)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            return result
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return {"error": str(e), "success": False}
        
        finally:
            if 'repo_path' in locals():
                shutil.rmtree(repo_path, ignore_errors=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current analyzer status"""
        gemini_available = self.gemini_service and self.gemini_service.is_available()
        return {
            "gemini_available": gemini_available,
            "ollama_available": self.ollama_available,
            "api_requests_made": self.api_requests_made,
            "api_limit_reached": not gemini_available and self.gemini_service is not None,
            "recommended_approach": self._get_recommended_approach(),
            "has_any_llm": gemini_available or self.ollama_available,
            "gemini_service_status": self.gemini_service.get_service_name() if self.gemini_service else "Not initialized"
        }
    
    def _get_recommended_approach(self) -> str:
        """Get recommendation based on current status"""
        gemini_available = self.gemini_service and self.gemini_service.is_available()
        if not gemini_available and self.gemini_service is not None:
            return "Local Ollama (API quota reached)"
        elif gemini_available:
            return "Gemini API (Fast)"
        elif self.ollama_available:
            return "Local Ollama (Slower)"
        else:
            return "Neither available"


# Initialize analyzer with automatic configuration
@st.cache_resource
def get_analyzer():
    # Auto-detect Gemini API keys from secrets
    gemini_api_keys = []
    try:
        for i in range(1, 5):
            key_name = f"gemini_api_key_{i}"
            if hasattr(st.secrets, key_name):
                api_key = getattr(st.secrets, key_name)
                if api_key:
                    gemini_api_keys.append(api_key)
    except:
        pass
    
    return RepositoryAnalyzer(gemini_api_keys=gemini_api_keys if gemini_api_keys else None)

analyzer = get_analyzer()

# How to Use section
with st.expander("How to Use"):
    st.info("""
    **Simple Steps:**
    1. Paste any GitHub repository URL below
    2. Click Analyze
    3. View results in the tabs that appear
    """)

st.markdown("---")
repo_url = st.text_input("Repository URL", placeholder="https://github.com/username/repository")

col1, col2 = st.columns([3, 1])
with col1:
    analyze_button = st.button("Analyze", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear", use_container_width=True)

# Clear results if requested
if clear_button:
    if 'analysis_results' in st.session_state:
        del st.session_state['analysis_results']
    st.rerun()

# Perform analysis
if analyze_button and repo_url:
    status = analyzer.get_status()
    if not status["has_any_llm"]:
        st.error("No AI service available.")
        st.stop()
    
    start_time = time.time()
    
    try:
        result = analyzer.analyze_repository_progressive(repo_url)
        end_time = time.time()
        analysis_time = end_time - start_time
        
        if result and result.get('success'):
            st.session_state['analysis_results'] = result
            st.session_state['analysis_time'] = analysis_time
            st.session_state['current_repo_url'] = repo_url
            st.success(f"Completed in {analysis_time:.1f}s")
        else:
            st.error(f"Failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"Failed: {str(e)}")

# Display results
if 'analysis_results' in st.session_state:
    result = st.session_state['analysis_results']
    
    if result.get('success', False):
        # Repository name
        st.subheader(result['repository'])
        
        analysis = result['analysis']
        quality_metrics = analysis.get('quality_metrics', {})
        
        # Simple metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Files", analysis["file_structure"]["total_files"])
        with col2:
            st.metric("Lines", f"{analysis['file_structure']['total_lines']:,}")
        with col3:
            st.metric("Technologies", len(analysis["technologies"]))
        with col4:
            overall_score = quality_metrics.get('overall_score', {})
            score = overall_score.get('score', 0)
            st.metric("Quality", f"{score}/100")
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Files", "Charts", "Chat"])
        
        with tab1:
            
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
            
            file_explanations = analysis.get('file_explanations', {})
            
            if file_explanations:
                # Search functionality
                search_term = st.text_input("Search files:", placeholder="Type to filter files...")
                
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
                    
                    # Display files
                    for file_path, explanation in filtered_files.items():
                        with st.expander(file_path):
                            st.markdown(explanation)
            else:
                st.info("File explanations not available. Configure Gemini API key or make sure Ollama is running and re-run the analysis.")
        
        with tab3:
            
            insights = result.get('insights', '')
            if insights:
                st.markdown(insights)
            else:
                st.info("AI insights not available. Configure Gemini API key or make sure Ollama is running.")
        
        with tab4:
            
            # Initialize mentor in session state
            if 'code_mentor' not in st.session_state:
                if 'analysis_results' in st.session_state:
                    # Instantiate mentor with analyzer instance
                    st.session_state.code_mentor = InteractiveCodeMentor(analyzer)
                    # Provide the analysis context to the mentor
                    st.session_state.code_mentor.set_analysis_context(
                        st.session_state['analysis_results']
                    )
                    # Initialize empty conversation history
                    st.session_state.mentor_messages = []
                else:
                    st.warning("Please analyze a repository first to start the Code Mentor.")
                    st.stop()
            
            # Display conversation history
            for message in st.session_state.get('mentor_messages', []):
                # Handle both dictionary and string message formats
                if isinstance(message, dict) and message.get('message'):
                    message_content = message['message']
                elif isinstance(message, str):
                    message_content = message
                    message = {'message': message_content}  # Convert to dict format
                else:
                    continue  # Skip invalid messages
                
                with st.chat_message("assistant" if 'suggestions' in str(message) else "user"):
                    st.write(message_content)
                    
                    # Show diagram if available
                    if isinstance(message, dict) and message.get('diagram'):
                        render_mermaid_diagram(
                            message['diagram'],
                            "Explanation Diagram",
                            "Visual representation of the concept",
                            key_suffix=f"mentor_{len(st.session_state.mentor_messages)}"
                        )
                    
                    # Show code examples if available
                    if isinstance(message, dict) and message.get('code_examples'):
                        st.markdown("**Relevant Code Examples:**")
                        for example in message['code_examples']:
                            st.code(f"# {example['description']}\n# See: {example['file']}")
                    
                    # Show exercise if created
                    if isinstance(message, dict) and message.get('exercise'):
                        exercise = message['exercise']
                        with st.expander("Exercise Details"):
                            st.write(exercise['content'])
                            
                            # Setup commands
                            if message.get('setup_commands'):
                                st.markdown("**Setup Commands:**")
                                st.code('\n'.join(message['setup_commands']))
                            
                            # Hint system
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("Hint 1", key=f"hint1_{exercise['id']}"):
                                    hint = asyncio.run(
                                        st.session_state.code_mentor.provide_hint(exercise['id'], 1)
                                    )
                                    st.info(hint)
                            with col2:
                                if st.button("Hint 2", key=f"hint2_{exercise['id']}"):
                                    hint = asyncio.run(
                                        st.session_state.code_mentor.provide_hint(exercise['id'], 2)
                                    )
                                    st.info(hint)
                            with col3:
                                if st.button("Hint 3", key=f"hint3_{exercise['id']}"):
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
                                            st.success("Exercise completed successfully!")
                                        
                                        # Next steps
                                        if result['next_steps']:
                                            st.markdown("**Next Steps:**")
                                            for step in result['next_steps']:
                                                st.write(f"• {step}")
            
            # Suggestions
            if st.session_state.get('mentor_messages'):
                last_message = st.session_state.mentor_messages[-1]
                if last_message.get('follow_up_suggestions') or last_message.get('suggestions'):
                    st.markdown("**Suggested Questions:**")
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
                    st.session_state.mentor_messages.append({'message': response})
                
                # Rerun to display new messages
                st.rerun()
            
            # Learning progress sidebar
            with st.sidebar:
                st.markdown("### Learning Progress")
                
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
                        st.markdown("**Completed Exercises:**")
                        for exercise in completed:
                            st.write(f"• {exercise['topic']} ({exercise['difficulty']})")
