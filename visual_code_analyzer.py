# visual_code_analyzer.py - Generate visual representations of code flow and architecture

import ast
import re
import json
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, deque
from pathlib import Path

class VisualCodeAnalyzer:
    """
    Generate visual diagrams and flow representations of code.
    
    This analyzer creates multiple types of visual representations:
    1. System Flow Diagrams - showing how the main application flows
    2. Component Interaction Maps - how different parts interact
    3. Data Flow Visualization - how data moves through the system
    4. Function Call Hierarchies - which functions call which others
    5. Class Relationship Diagrams - inheritance and composition
    
    Think of this as creating a visual "blueprint" of how the code works,
    like an architect's blueprint shows how a building is constructed.
    """
    
    def __init__(self):
        self.main_flows = []  # Main application entry points and flows
        self.component_interactions = {}  # How components interact
        self.data_flows = []  # How data moves through the system
        self.function_hierarchy = defaultdict(set)  # Function call relationships
        self.class_relationships = defaultdict(set)  # Class inheritance/composition
        self.entry_points = []  # Main entry points (main functions, API endpoints, etc.)
    
    def analyze_code_visually(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for visual code analysis.
        
        Analyzes the codebase and generates multiple visual representations
        showing how the code works and flows.
        """
        visual_report = {
            'system_overview': {},
            'main_flows': [],
            'component_interactions': [],
            'data_flows': [],
            'entry_points': [],
            'visual_diagrams': {
                'system_flow': None,
                'component_interaction': None,
                'data_flow': None,
                'function_hierarchy': None
            },
            'insights': [],
            'recommendations': []
        }
        
        # Step 1: Identify entry points and main functions
        visual_report['entry_points'] = self._identify_entry_points(detailed_files)
        
        # Step 2: Analyze system flow from entry points
        visual_report['main_flows'] = self._analyze_main_flows(detailed_files, visual_report['entry_points'])
        
        # Step 3: Map component interactions
        visual_report['component_interactions'] = self._analyze_component_interactions(detailed_files)
        
        # Step 4: Trace data flows
        visual_report['data_flows'] = self._analyze_data_flows(detailed_files)
        
        # Step 5: Generate visual diagrams
        visual_report['visual_diagrams'] = self._generate_visual_diagrams(visual_report)
        
        # Step 6: Generate insights about the codebase structure
        visual_report['insights'] = self._generate_visual_insights(visual_report, detailed_files)
        
        # Step 7: Provide recommendations for better structure
        visual_report['recommendations'] = self._generate_visual_recommendations(visual_report)
        
        # Step 8: Create system overview
        visual_report['system_overview'] = self._create_system_overview(detailed_files, visual_report)
        
        return visual_report
    
    def _identify_entry_points(self, detailed_files: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify main entry points in the codebase.
        
        Entry points are like the "front doors" of your application:
        - main() functions
        - if __name__ == "__main__": blocks
        - API endpoints (Flask/FastAPI routes)
        - CLI command handlers
        - Web server startup files
        """
        entry_points = []
        
        for file_path, file_info in detailed_files.items():
            if not file_path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                continue
            
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            if not content:
                continue
            
            # Python entry points
            if file_path.endswith('.py'):
                entry_points.extend(self._find_python_entry_points(file_path, content))
            
            # JavaScript/TypeScript entry points
            elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                entry_points.extend(self._find_javascript_entry_points(file_path, content))
        
        return entry_points
    
    def _find_python_entry_points(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Find Python entry points like main() functions and __main__ blocks."""
        entry_points = []
        
        # Look for if __name__ == "__main__": blocks
        if 'if __name__ == "__main__"' in content:
            entry_points.append({
                'file': file_path,
                'type': 'main_script',
                'name': f"{Path(file_path).stem} (main script)",
                'description': 'Script entry point',
                'line': self._find_line_number(content, 'if __name__ == "__main__"')
            })
        
        # Look for main() functions
        main_function_patterns = [
            r'def main\s*\(',
            r'def run\s*\(',
            r'def start\s*\(',
            r'def execute\s*\('
        ]
        
        for pattern in main_function_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                func_name = match.group(0).split('def ')[1].split('(')[0].strip()
                entry_points.append({
                    'file': file_path,
                    'type': 'main_function',
                    'name': f"{func_name}()",
                    'description': f'Main function in {Path(file_path).name}',
                    'line': self._find_line_number(content, match.group(0))
                })
        
        # Look for Flask/FastAPI routes
        route_patterns = [
            r'@app\.route\s*\(',
            r'@router\.get\s*\(',
            r'@router\.post\s*\(',
            r'@app\.get\s*\(',
            r'@app\.post\s*\('
        ]
        
        for pattern in route_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entry_points.append({
                    'file': file_path,
                    'type': 'api_endpoint',
                    'name': f"API endpoint in {Path(file_path).name}",
                    'description': 'Web API endpoint',
                    'line': self._find_line_number(content, match.group(0))
                })
        
        # Look for Streamlit apps
        if 'streamlit' in content.lower() and 'st.' in content:
            entry_points.append({
                'file': file_path,
                'type': 'streamlit_app',
                'name': f"Streamlit app ({Path(file_path).name})",
                'description': 'Streamlit web application',
                'line': 1
            })
        
        return entry_points
    
    def _find_javascript_entry_points(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Find JavaScript/TypeScript entry points."""
        entry_points = []
        
        # Look for main/index files
        if Path(file_path).stem.lower() in ['index', 'main', 'app', 'server']:
            entry_points.append({
                'file': file_path,
                'type': 'main_script',
                'name': f"{Path(file_path).name} (entry script)",
                'description': 'Main application entry point',
                'line': 1
            })
        
        # Look for Express.js routes
        express_patterns = [
            r'app\.get\s*\(',
            r'app\.post\s*\(',
            r'router\.get\s*\(',
            r'router\.post\s*\('
        ]
        
        for pattern in express_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entry_points.append({
                    'file': file_path,
                    'type': 'api_endpoint',
                    'name': f"API route in {Path(file_path).name}",
                    'description': 'Express.js API endpoint',
                    'line': self._find_line_number(content, match.group(0))
                })
        
        # Look for React components
        if 'react' in content.lower() and ('function ' in content or 'const ' in content and '=>' in content):
            entry_points.append({
                'file': file_path,
                'type': 'react_component',
                'name': f"React component ({Path(file_path).stem})",
                'description': 'React UI component',
                'line': 1
            })
        
        return entry_points
    
    def _analyze_main_flows(self, detailed_files: Dict[str, Any], entry_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze the main execution flows starting from entry points.
        
        This traces how the application flows from start to finish,
        like following the path a user takes through a building.
        """
        main_flows = []
        
        for entry_point in entry_points[:5]:  # Analyze top 5 entry points
            flow = {
                'entry_point': entry_point,
                'flow_steps': [],
                'components_involved': [],
                'data_transformations': []
            }
            
            # Get the file content for this entry point
            file_info = detailed_files.get(entry_point['file'])
            if not file_info:
                continue
            
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            if not content:
                continue
            
            # Trace the flow from this entry point
            flow['flow_steps'] = self._trace_execution_flow(content, entry_point, detailed_files)
            flow['components_involved'] = self._identify_components_in_flow(flow['flow_steps'])
            flow['data_transformations'] = self._identify_data_transformations(flow['flow_steps'])
            
            main_flows.append(flow)
        
        return main_flows
    
    def _trace_execution_flow(self, content: str, entry_point: Dict[str, Any], detailed_files: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Trace the execution flow from an entry point.
        
        This is like following a recipe step by step to see what happens.
        """
        flow_steps = []
        
        # Start with the entry point
        flow_steps.append({
            'step': 1,
            'action': f"Start at {entry_point['name']}",
            'type': 'entry',
            'file': entry_point['file'],
            'description': entry_point['description']
        })
        
        # Extract function calls and imports
        functions_called = self._extract_function_calls(content)
        imports = self._extract_imports(content)
        
        step_num = 2
        
        # Add import steps
        for imp in imports[:5]:  # Limit to top 5 imports
            flow_steps.append({
                'step': step_num,
                'action': f"Import {imp}",
                'type': 'import',
                'description': f"Load external functionality from {imp}"
            })
            step_num += 1
        
        # Add function call steps
        for func in functions_called[:8]:  # Limit to top 8 function calls
            flow_steps.append({
                'step': step_num,
                'action': f"Call {func}",
                'type': 'function_call',
                'description': f"Execute {func} function"
            })
            step_num += 1
        
        # Add conditional logic detection
        conditionals = self._extract_conditional_logic(content)
        for cond in conditionals[:3]:  # Limit to top 3 conditionals
            flow_steps.append({
                'step': step_num,
                'action': f"Decision point: {cond}",
                'type': 'conditional',
                'description': f"Branch based on {cond}"
            })
            step_num += 1
        
        return flow_steps
    
    def _analyze_component_interactions(self, detailed_files: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze how different components/modules interact with each other.
        
        This is like mapping out how different departments in a company
        work together to get things done.
        """
        interactions = []
        component_map = {}
        
        # First, identify all components
        for file_path, file_info in detailed_files.items():
            if not file_path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                continue
            
            component_name = Path(file_path).stem
            component_map[component_name] = {
                'file': file_path,
                'type': self._determine_component_type(file_path, file_info),
                'exports': self._extract_exports(file_info.get('full_content', '')),
                'imports': self._extract_imports(file_info.get('full_content', ''))
            }
        
        # Then, analyze interactions between components
        for comp_name, comp_info in component_map.items():
            for imported_module in comp_info['imports']:
                # Check if this is a local import (component interaction)
                if imported_module in component_map:
                    interactions.append({
                        'from_component': comp_name,
                        'to_component': imported_module,
                        'interaction_type': 'import',
                        'description': f"{comp_name} uses functionality from {imported_module}"
                    })
        
        return interactions
    
    def _analyze_data_flows(self, detailed_files: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze how data flows through the system.
        
        This traces data like water flowing through pipes,
        showing where it comes from, how it's transformed, and where it goes.
        """
        data_flows = []
        
        for file_path, file_info in detailed_files.items():
            if not file_path.endswith(('.py', '.js', '.ts')):
                continue
            
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            if not content:
                continue
            
            # Look for data input sources
            data_inputs = self._identify_data_inputs(content)
            
            # Look for data transformations
            transformations = self._identify_data_transformations_in_file(content)
            
            # Look for data outputs
            data_outputs = self._identify_data_outputs(content)
            
            if data_inputs or transformations or data_outputs:
                data_flows.append({
                    'file': file_path,
                    'inputs': data_inputs,
                    'transformations': transformations,
                    'outputs': data_outputs
                })
        
        return data_flows
    
    def _generate_visual_diagrams(self, visual_report: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate Mermaid diagrams for different visualizations.
        
        Creates interactive diagrams that show the visual structure of the code.
        """
        diagrams = {}
        
        # Generate system flow diagram
        diagrams['system_flow'] = self._create_system_flow_diagram(visual_report)
        
        # Generate component interaction diagram
        diagrams['component_interaction'] = self._create_component_interaction_diagram(visual_report)
        
        # Generate data flow diagram
        diagrams['data_flow'] = self._create_data_flow_diagram(visual_report)
        
        return diagrams
    
    def _create_system_flow_diagram(self, visual_report: Dict[str, Any]) -> str:
        """Create a Mermaid diagram showing the main system flow."""
        if not visual_report['main_flows']:
            return "graph TD\n    A[\"No main flows detected\"]"
        
        # Take the first main flow
        main_flow = visual_report['main_flows'][0]
        flow_steps = main_flow['flow_steps']
        
        mermaid_lines = ["graph TD"]
        
        for i, step in enumerate(flow_steps):
            step_id = f"S{step['step']}"
            step_label = step['action'].replace('"', "'")
            
            if step['type'] == 'entry':
                mermaid_lines.append(f'    {step_id}["{step_label}"]')
                mermaid_lines.append(f'    style {step_id} fill:#e1f5fe')
            elif step['type'] == 'import':
                mermaid_lines.append(f'    {step_id}["{step_label}"]')
                mermaid_lines.append(f'    style {step_id} fill:#f3e5f5')
            elif step['type'] == 'function_call':
                mermaid_lines.append(f'    {step_id}["{step_label}"]')
                mermaid_lines.append(f'    style {step_id} fill:#e8f5e8')
            elif step['type'] == 'conditional':
                mermaid_lines.append(f'    {step_id}{{{step_label}}}')
                mermaid_lines.append(f'    style {step_id} fill:#fff3e0')
            
            # Connect to next step
            if i < len(flow_steps) - 1:
                next_step_id = f"S{flow_steps[i+1]['step']}"
                mermaid_lines.append(f'    {step_id} --> {next_step_id}')
        
        return "\n".join(mermaid_lines)
    
    def _create_component_interaction_diagram(self, visual_report: Dict[str, Any]) -> str:
        """Create a Mermaid diagram showing component interactions."""
        interactions = visual_report['component_interactions']
        
        if not interactions:
            return "graph TD\n    A[\"No component interactions detected\"]"
        
        mermaid_lines = ["graph TD"]
        added_components = set()
        
        for interaction in interactions[:10]:  # Limit to first 10 interactions
            from_comp = interaction['from_component'].replace(' ', '_').replace('.', '_')
            to_comp = interaction['to_component'].replace(' ', '_').replace('.', '_')
            
            # Add components if not already added
            if from_comp not in added_components:
                mermaid_lines.append(f'    {from_comp}["{interaction["from_component"]}"]')
                added_components.add(from_comp)
            
            if to_comp not in added_components:
                mermaid_lines.append(f'    {to_comp}["{interaction["to_component"]}"]')
                added_components.add(to_comp)
            
            # Add interaction
            mermaid_lines.append(f'    {from_comp} --> {to_comp}')
        
        return "\n".join(mermaid_lines)
    
    def _create_data_flow_diagram(self, visual_report: Dict[str, Any]) -> str:
        """Create a Mermaid diagram showing data flow."""
        data_flows = visual_report['data_flows']
        
        if not data_flows:
            return "graph TD\n    A[\"No data flows detected\"]"
        
        mermaid_lines = ["graph TD"]
        node_counter = 1
        
        for flow in data_flows[:5]:  # Limit to first 5 data flows
            file_name = Path(flow['file']).stem.replace(' ', '_').replace('.', '_')
            
            # Add inputs
            for inp in flow['inputs'][:3]:  # Limit to 3 inputs per file
                input_id = f"I{node_counter}"
                mermaid_lines.append(f'    {input_id}["{inp}"]')
                mermaid_lines.append(f'    style {input_id} fill:#e3f2fd')
                mermaid_lines.append(f'    {input_id} --> {file_name}')
                node_counter += 1
            
            # Add the processing file
            mermaid_lines.append(f'    {file_name}["{Path(flow["file"]).stem}"]')
            mermaid_lines.append(f'    style {file_name} fill:#f1f8e9')
            
            # Add outputs
            for out in flow['outputs'][:3]:  # Limit to 3 outputs per file
                output_id = f"O{node_counter}"
                mermaid_lines.append(f'    {output_id}["{out}"]')
                mermaid_lines.append(f'    style {output_id} fill:#fce4ec')
                mermaid_lines.append(f'    {file_name} --> {output_id}')
                node_counter += 1
        
        return "\n".join(mermaid_lines)
    
    # Helper methods
    def _find_line_number(self, content: str, search_text: str) -> int:
        """Find the line number of a specific text in content."""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if search_text in line:
                return i
        return 1
    
    def _extract_function_calls(self, content: str) -> List[str]:
        """Extract function calls from code content."""
        # Simple regex to find function calls
        pattern = r'(\w+)\s*\('
        matches = re.findall(pattern, content)
        
        # Filter out common keywords and duplicates
        keywords_to_exclude = {'if', 'for', 'while', 'def', 'class', 'import', 'from', 'return', 'print'}
        function_calls = []
        seen = set()
        
        for match in matches:
            if match not in keywords_to_exclude and match not in seen:
                function_calls.append(match)
                seen.add(match)
        
        return function_calls[:10]  # Return top 10
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('import '):
                module = line.replace('import ', '').split(' as ')[0].split('.')[0]
                imports.append(module)
            elif line.startswith('from '):
                if ' import ' in line:
                    module = line.split('from ')[1].split(' import ')[0].split('.')[0]
                    imports.append(module)
        
        return list(set(imports))[:8]  # Return unique imports, max 8
    
    def _extract_conditional_logic(self, content: str) -> List[str]:
        """Extract conditional logic from code."""
        conditionals = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('if ') and ':' in line:
                condition = line.replace('if ', '').replace(':', '').strip()
                if len(condition) < 50:  # Only short conditions
                    conditionals.append(condition)
        
        return conditionals[:5]
    
    def _determine_component_type(self, file_path: str, file_info: Dict[str, Any]) -> str:
        """Determine the type of component based on file path and content."""
        content = file_info.get('full_content', '').lower()
        file_name = Path(file_path).name.lower()
        
        if 'test' in file_name:
            return 'test'
        elif 'config' in file_name:
            return 'configuration'
        elif 'util' in file_name or 'helper' in file_name:
            return 'utility'
        elif 'api' in file_name or '@app.route' in content:
            return 'api'
        elif 'model' in file_name or 'class ' in content:
            return 'model'
        elif 'view' in file_name or 'render' in content:
            return 'view'
        else:
            return 'module'
    
    def _extract_exports(self, content: str) -> List[str]:
        """Extract what a module exports (functions, classes, etc.)."""
        exports = []
        
        # Python exports (functions and classes defined)
        function_pattern = r'def\s+(\w+)\s*\('
        class_pattern = r'class\s+(\w+)\s*[\(:]'
        
        functions = re.findall(function_pattern, content)
        classes = re.findall(class_pattern, content)
        
        exports.extend(functions[:5])  # Max 5 functions
        exports.extend(classes[:3])    # Max 3 classes
        
        return exports
    
    def _identify_data_inputs(self, content: str) -> List[str]:
        """Identify data input sources in code."""
        inputs = []
        
        # Common data input patterns
        input_patterns = [
            r'open\s*\(\s*["\']([^"\']+)["\']',  # File reading
            r'read_csv\s*\(\s*["\']([^"\']+)["\']',  # CSV reading
            r'\.json\(\)',  # JSON parsing
            r'input\s*\(',  # User input
            r'request\.',  # Web request data
            r'sys\.argv',  # Command line arguments
        ]
        
        for pattern in input_patterns:
            matches = re.findall(pattern, content)
            if pattern == r'\.json\(\)':
                inputs.append('JSON data')
            elif pattern == r'input\s*\(':
                inputs.append('User input')
            elif pattern == r'request\.':
                inputs.append('HTTP request data')
            elif pattern == r'sys\.argv':
                inputs.append('Command line arguments')
            else:
                inputs.extend(matches[:2])  # Max 2 per pattern
        
        return inputs[:5]  # Max 5 total
    
    def _identify_data_transformations_in_file(self, content: str) -> List[str]:
        """Identify data transformations in a file."""
        transformations = []
        
        # Common transformation patterns
        if '.map(' in content:
            transformations.append('Data mapping/transformation')
        if '.filter(' in content:
            transformations.append('Data filtering')
        if '.sort(' in content or 'sorted(' in content:
            transformations.append('Data sorting')
        if '.groupby(' in content or 'group_by' in content:
            transformations.append('Data grouping')
        if 'json.loads(' in content or 'json.dumps(' in content:
            transformations.append('JSON processing')
        if 'str.replace(' in content or '.replace(' in content:
            transformations.append('String processing')
        
        return transformations[:4]  # Max 4
    
    def _identify_data_outputs(self, content: str) -> List[str]:
        """Identify data output destinations in code."""
        outputs = []
        
        # Common output patterns
        output_patterns = [
            r'\.to_csv\s*\(',  # CSV writing
            r'\.to_json\s*\(',  # JSON writing
            r'print\s*\(',  # Console output
            r'return\s+',  # Function return
            r'render_template\s*\(',  # Web template rendering
            r'jsonify\s*\(',  # JSON response
        ]
        
        for pattern in output_patterns:
            if re.search(pattern, content):
                if 'to_csv' in pattern:
                    outputs.append('CSV file')
                elif 'to_json' in pattern:
                    outputs.append('JSON file')
                elif 'print' in pattern:
                    outputs.append('Console output')
                elif 'return' in pattern:
                    outputs.append('Function result')
                elif 'render_template' in pattern:
                    outputs.append('Web page')
                elif 'jsonify' in pattern:
                    outputs.append('JSON response')
        
        return outputs[:4]  # Max 4
    
    def _identify_components_in_flow(self, flow_steps: List[Dict[str, Any]]) -> List[str]:
        """Identify components involved in a flow."""
        components = set()
        for step in flow_steps:
            if step['type'] == 'import':
                components.add(step['action'].replace('Import ', ''))
            elif 'file' in step:
                components.add(Path(step['file']).stem)
        return list(components)[:8]  # Max 8 components
    
    def _identify_data_transformations(self, flow_steps: List[Dict[str, Any]]) -> List[str]:
        """Identify data transformations in a flow."""
        transformations = []
        for step in flow_steps:
            if 'transform' in step['action'].lower() or 'process' in step['action'].lower():
                transformations.append(step['action'])
        return transformations[:5]  # Max 5
    
    def _generate_visual_insights(self, visual_report: Dict[str, Any], detailed_files: Dict[str, Any]) -> List[str]:
        """Generate insights about the visual structure of the codebase."""
        insights = []
        
        entry_points = visual_report['entry_points']
        main_flows = visual_report['main_flows']
        interactions = visual_report['component_interactions']
        
        # Insights about entry points
        if len(entry_points) == 1:
            insights.append(f"🎯 Single entry point detected: {entry_points[0]['name']} - Simple, focused application")
        elif len(entry_points) > 5:
            insights.append(f"🔀 Multiple entry points ({len(entry_points)}) - Complex application with many access points")
        
        # Insights about flow complexity
        if main_flows:
            avg_steps = sum(len(flow['flow_steps']) for flow in main_flows) / len(main_flows)
            if avg_steps > 10:
                insights.append(f"🌊 Complex execution flows detected (avg {avg_steps:.1f} steps) - Consider simplification")
            else:
                insights.append(f"✅ Simple execution flows (avg {avg_steps:.1f} steps) - Good maintainability")
        
        # Insights about component interactions
        if len(interactions) > 20:
            insights.append(f"🕸️ High component coupling ({len(interactions)} interactions) - Consider modularization")
        elif len(interactions) < 5:
            insights.append("🏗️ Low component coupling - Well-isolated modules")
        
        # Technology-specific insights
        total_files = len(detailed_files)
        python_files = len([f for f in detailed_files.keys() if f.endswith('.py')])
        js_files = len([f for f in detailed_files.keys() if f.endswith(('.js', '.jsx', '.ts', '.tsx'))])
        
        if python_files > js_files * 2:
            insights.append("🐍 Python-heavy codebase - Backend/data processing focus")
        elif js_files > python_files * 2:
            insights.append("🌐 JavaScript-heavy codebase - Frontend/web focus")
        
        return insights[:6]  # Max 6 insights
    
    def _generate_visual_recommendations(self, visual_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving code structure."""
        recommendations = []
        
        entry_points = visual_report['entry_points']
        interactions = visual_report['component_interactions']
        
        # Recommendations based on entry points
        if len(entry_points) == 0:
            recommendations.append("🎯 Add clear entry points (main functions) to improve code organization")
        elif len(entry_points) > 8:
            recommendations.append("🔀 Consider consolidating entry points - too many can confuse users")
        
        # Recommendations based on interactions
        if len(interactions) > 25:
            recommendations.append("🏗️ High coupling detected - consider breaking into smaller, focused modules")
        
        # Flow-based recommendations
        main_flows = visual_report['main_flows']
        if main_flows:
            for flow in main_flows:
                if len(flow['flow_steps']) > 15:
                    recommendations.append("📏 Some execution flows are very long - consider breaking into smaller functions")
                    break
        
        # General recommendations
        recommendations.append("📊 Use the visual diagrams to identify refactoring opportunities")
        recommendations.append("🔍 Look for components with too many connections - they might need splitting")
        
        return recommendations[:5]  # Max 5 recommendations
    
    def _create_system_overview(self, detailed_files: Dict[str, Any], visual_report: Dict[str, Any]) -> Dict[str, Any]:
        """Create a high-level overview of the system."""
        total_files = len(detailed_files)
        code_files = len([f for f in detailed_files.keys() if f.endswith(('.py', '.js', '.ts', '.jsx', '.tsx'))])
        
        return {
            'total_files': total_files,
            'code_files': code_files,
            'entry_points': len(visual_report['entry_points']),
            'main_flows': len(visual_report['main_flows']),
            'component_interactions': len(visual_report['component_interactions']),
            'complexity_level': self._assess_complexity_level(visual_report),
            'architecture_style': self._determine_architecture_style(visual_report, detailed_files)
        }
    
    def _assess_complexity_level(self, visual_report: Dict[str, Any]) -> str:
        """Assess the overall complexity level of the system."""
        entry_points = len(visual_report['entry_points'])
        interactions = len(visual_report['component_interactions'])
        flows = len(visual_report['main_flows'])
        
        complexity_score = entry_points + interactions + flows
        
        if complexity_score < 10:
            return 'Simple'
        elif complexity_score < 25:
            return 'Moderate'
        elif complexity_score < 50:
            return 'Complex'
        else:
            return 'Very Complex'
    
    def _determine_architecture_style(self, visual_report: Dict[str, Any], detailed_files: Dict[str, Any]) -> str:
        """Determine the architectural style of the codebase."""
        file_names = [Path(f).name.lower() for f in detailed_files.keys()]
        
        # Check for common architectural patterns
        if any('api' in name for name in file_names) and any('model' in name for name in file_names):
            return 'API-based / MVC-like'
        elif any('component' in name for name in file_names) or any('react' in str(detailed_files.values())):
            return 'Component-based'
        elif len(visual_report['entry_points']) == 1:
            return 'Monolithic script'
        elif 'app.py' in file_names or 'main.py' in file_names:
            return 'Application-centric'
        else:
            return 'Modular' 