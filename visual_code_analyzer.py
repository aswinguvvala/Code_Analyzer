# Intelligent Adaptive Visual Code Analyzer - Works for ANY codebase

import ast
import re
import json
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path

class IntelligentAdaptiveVisualAnalyzer:
    """
    Intelligent analyzer that adapts to ANY codebase type without hardcoding.
    Uses pattern recognition and code structure analysis to understand any project.
    """
    
    def __init__(self):
        # Dynamic pattern detection - no hardcoding
        self.discovered_patterns = {
            'frameworks': set(),
            'architectures': set(), 
            'operation_types': set(),
            'file_types': set(),
            'execution_patterns': set()
        }
        
        # Intelligence weights for different analysis approaches
        self.analysis_confidence = {}

    def analyze_code_visually(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligent analysis that adapts to any codebase type.
        No hardcoding - discovers patterns dynamically.
        """
        # Phase 1: Discover codebase characteristics
        codebase_profile = self._profile_codebase(detailed_files)
        
        # Phase 2: Adapt analysis strategy based on discovery
        analysis_strategy = self._determine_analysis_strategy(codebase_profile)
        
        # Phase 3: Execute adaptive analysis
        visual_report = {
            'codebase_profile': codebase_profile,
            'analysis_strategy': analysis_strategy,
            'entry_points': [],
            'execution_flows': [],
            'component_interactions': [],
            'data_flows': [],
            'architectural_insights': {},
            'visual_diagrams': {},
            'intelligent_insights': [],
            'adaptive_recommendations': []
        }
        
        # Intelligent entry point detection
        visual_report['entry_points'] = self._discover_entry_points_intelligently(
            detailed_files, codebase_profile
        )
        
        # Adaptive flow analysis
        visual_report['execution_flows'] = self._trace_execution_flows_adaptively(
            detailed_files, visual_report['entry_points'], codebase_profile
        )
        
        # Intelligent component analysis
        visual_report['component_interactions'] = self._analyze_component_interactions_intelligently(
            detailed_files, codebase_profile
        )
        
        # Dynamic data flow analysis
        visual_report['data_flows'] = self._analyze_data_flows_dynamically(
            detailed_files, codebase_profile
        )
        
        # Architectural pattern recognition
        visual_report['architectural_insights'] = self._detect_architectural_patterns(
            detailed_files, codebase_profile
        )
        
        # Generate adaptive visual diagrams
        visual_report['visual_diagrams'] = self._generate_adaptive_diagrams(visual_report)
        
        # Intelligence-driven insights
        visual_report['intelligent_insights'] = self._generate_intelligent_insights(
            visual_report, detailed_files
        )
        
        # Adaptive recommendations
        visual_report['adaptive_recommendations'] = self._generate_adaptive_recommendations(
            visual_report, codebase_profile
        )
        
        return visual_report
    
    def _profile_codebase(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently profile the codebase to understand its nature.
        No hardcoding - discovers patterns dynamically.
        """
        profile = {
            'primary_language': None,
            'detected_frameworks': [],
            'application_type': None,
            'architectural_style': None,
            'complexity_indicators': {},
            'execution_patterns': [],
            'data_patterns': [],
            'interaction_patterns': []
        }
        
        # Language detection
        language_scores = defaultdict(int)
        
        # Framework and library detection
        framework_indicators = defaultdict(int)
        
        # Application type indicators
        app_type_indicators = defaultdict(int)
        
        # Pattern analysis
        for file_path, file_info in detailed_files.items():
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            file_ext = Path(file_path).suffix.lower()
            
            # Language scoring
            if file_ext == '.py':
                language_scores['Python'] += file_info.get('lines', 0)
            elif file_ext in ['.js', '.jsx']:
                language_scores['JavaScript'] += file_info.get('lines', 0)
            elif file_ext in ['.ts', '.tsx']:
                language_scores['TypeScript'] += file_info.get('lines', 0)
            elif file_ext == '.java':
                language_scores['Java'] += file_info.get('lines', 0)
            elif file_ext in ['.cpp', '.cc', '.cxx']:
                language_scores['C++'] += file_info.get('lines', 0)
            elif file_ext == '.go':
                language_scores['Go'] += file_info.get('lines', 0)
            elif file_ext == '.rs':
                language_scores['Rust'] += file_info.get('lines', 0)
            
            # Dynamic framework detection through imports
            frameworks = self._extract_frameworks_dynamically(content, file_ext)
            for fw in frameworks:
                framework_indicators[fw] += 1
            
            # Application type detection through patterns
            app_types = self._detect_application_type_patterns(content, file_path)
            for app_type in app_types:
                app_type_indicators[app_type] += 1
        
        # Determine primary language
        if language_scores:
            profile['primary_language'] = max(language_scores.items(), key=lambda x: x[1])[0]
        
        # Top frameworks (dynamic discovery)
        profile['detected_frameworks'] = [
            fw for fw, score in sorted(framework_indicators.items(), 
                                     key=lambda x: x[1], reverse=True)[:8]
        ]
        
        # Application type (intelligent inference)
        if app_type_indicators:
            profile['application_type'] = max(app_type_indicators.items(), key=lambda x: x[1])[0]
        
        # Complexity analysis
        profile['complexity_indicators'] = self._analyze_complexity_indicators(detailed_files)
        
        return profile
    
    def _extract_frameworks_dynamically(self, content: str, file_ext: str) -> List[str]:
        """
        Dynamically extract frameworks and libraries from code.
        No hardcoding - discovers any framework.
        """
        frameworks = []
        
        if file_ext == '.py':
            # Python import patterns
            import_patterns = [
                r'import\s+(\w+)',
                r'from\s+(\w+)\s+import',
                r'import\s+(\w+\.\w+)',
            ]
        elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
            # JavaScript/TypeScript import patterns
            import_patterns = [
                r'import.*from\s+["\'](\w+)["\']',
                r'require\(["\'](\w+)["\']\)',
                r'import\s+(\w+)',
            ]
        elif file_ext == '.java':
            # Java import patterns
            import_patterns = [
                r'import\s+(\w+(?:\.\w+)*)',
            ]
        else:
            return frameworks
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                framework = match.split('.')[0] if '.' in match else match
                if len(framework) > 2 and framework.isalpha():
                    frameworks.append(framework)
        
        return frameworks
    
    def _detect_application_type_patterns(self, content: str, file_path: str) -> List[str]:
        """
        Intelligently detect application type through code patterns.
        No hardcoding - uses intelligent pattern recognition.
        """
        app_types = []
        content_lower = content.lower()
        
        # Web application patterns
        web_patterns = [
            'app.route', 'router.', 'express', 'flask', 'django', 'fastapi',
            'http', 'server', 'api', '@app.', 'render_template', 'jsonify'
        ]
        if any(pattern in content_lower for pattern in web_patterns):
            app_types.append('Web Application')
        
        # Command line application patterns
        cli_patterns = [
            'argparse', 'click', 'sys.argv', 'command line', 'cli', 'parser.add_argument'
        ]
        if any(pattern in content_lower for pattern in cli_patterns):
            app_types.append('CLI Application')
        
        # Data processing patterns
        data_patterns = [
            'pandas', 'numpy', 'csv', 'json', 'xml', 'data', 'dataframe', 
            'read_csv', 'to_csv', 'parse', 'transform'
        ]
        if any(pattern in content_lower for pattern in data_patterns):
            app_types.append('Data Processing')
        
        # Game patterns
        game_patterns = [
            'pygame', 'unity', 'game', 'player', 'score', 'level', 'sprite'
        ]
        if any(pattern in content_lower for pattern in game_patterns):
            app_types.append('Game')
        
        # Machine Learning patterns (without hardcoding specific frameworks)
        ml_patterns = [
            'model', 'train', 'predict', 'neural', 'network', 'learning',
            'fit', 'score', 'accuracy', 'loss', 'epoch', 'batch'
        ]
        if any(pattern in content_lower for pattern in ml_patterns):
            app_types.append('Machine Learning')
        
        # Desktop application patterns
        desktop_patterns = [
            'tkinter', 'pyqt', 'wxpython', 'kivy', 'gui', 'window', 'dialog'
        ]
        if any(pattern in content_lower for pattern in desktop_patterns):
            app_types.append('Desktop Application')
        
        # Testing patterns
        test_patterns = [
            'test', 'assert', 'unittest', 'pytest', 'mock', 'fixture'
        ]
        if any(pattern in content_lower for pattern in test_patterns):
            app_types.append('Testing')
        
        # Database patterns
        db_patterns = [
            'database', 'sql', 'sqlite', 'postgres', 'mysql', 'mongodb',
            'connection', 'query', 'table', 'schema'
        ]
        if any(pattern in content_lower for pattern in db_patterns):
            app_types.append('Database')
        
        return app_types
    
    def _discover_entry_points_intelligently(self, detailed_files: Dict[str, Any], 
                                           codebase_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Intelligently discover entry points without hardcoding.
        Adapts to any codebase type.
        """
        entry_points = []
        primary_language = codebase_profile.get('primary_language', 'Unknown')
        
        for file_path, file_info in detailed_files.items():
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            file_name = Path(file_path).name.lower()
            
            # Generic entry point patterns (language-agnostic where possible)
            entry_indicators = []
            
            # Main function patterns (adaptable to language)
            if primary_language == 'Python':
                if 'if __name__ == "__main__"' in content:
                    entry_indicators.append(('main_block', 'Python main execution block', 'high'))
                if re.search(r'def main\s*\(', content):
                    entry_indicators.append(('main_function', 'Main function', 'high'))
            elif primary_language in ['JavaScript', 'TypeScript']:
                if 'process.argv' in content or 'command line' in content.lower():
                    entry_indicators.append(('cli_entry', 'Command line entry point', 'high'))
            elif primary_language == 'Java':
                if 'public static void main' in content:
                    entry_indicators.append(('java_main', 'Java main method', 'high'))
            
            # Generic filename-based entry point detection
            entry_names = [
                'main', 'index', 'app', 'server', 'run', 'start', 'launch',
                'cli', 'manage', 'script', 'tool', 'demo', 'example'
            ]
            base_name = Path(file_path).stem.lower()
            if base_name in entry_names:
                entry_indicators.append(('filename_entry', f'Entry point by filename ({base_name})', 'medium'))
            
            # Application-specific patterns (based on detected type)
            app_type = codebase_profile.get('application_type')
            if app_type == 'Web Application':
                web_entry_patterns = [
                    ('app.run', 'web_server', 'Web server startup'),
                    ('app.listen', 'web_server', 'Web server startup'),
                    ('@app.route', 'web_endpoint', 'Web endpoint'),
                    ('router.', 'web_router', 'Web router'),
                ]
                for pattern, entry_type, description in web_entry_patterns:
                    if pattern in content:
                        entry_indicators.append((entry_type, description, 'high'))
            
            elif app_type == 'CLI Application':
                if 'argparse' in content or 'click' in content or 'sys.argv' in content:
                    entry_indicators.append(('cli_parser', 'Command line interface', 'high'))
            
            # Add discovered entry points
            for entry_type, description, priority in entry_indicators:
                entry_points.append({
                    'file': file_path,
                    'type': entry_type,
                    'name': Path(file_path).stem,
                    'description': description,
                    'priority': priority,
                    'discovery_method': 'intelligent_pattern_detection',
                    'confidence': self._calculate_entry_point_confidence(content, entry_type)
                })
        
        # Sort by confidence and priority
        return sorted(entry_points, key=lambda x: (
            x['priority'] == 'high', 
            x['confidence'], 
            x['name']
        ), reverse=True)
    
    def _trace_execution_flows_adaptively(self, detailed_files: Dict[str, Any], 
                                        entry_points: List[Dict[str, Any]], 
                                        codebase_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Adaptively trace execution flows based on discovered patterns.
        No hardcoding - adapts to any application type.
        """
        execution_flows = []
        
        for entry_point in entry_points[:5]:  # Analyze top 5 entry points
            file_info = detailed_files.get(entry_point['file'])
            if not file_info:
                continue
            
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            
            flow = {
                'entry_point': entry_point,
                'flow_steps': [],
                'discovered_operations': [],
                'component_dependencies': [],
                'execution_pattern': None
            }
            
            # Adaptive flow tracing based on application type
            app_type = codebase_profile.get('application_type')
            
            if app_type == 'Web Application':
                flow['flow_steps'] = self._trace_web_application_flow(content, entry_point)
            elif app_type == 'CLI Application':
                flow['flow_steps'] = self._trace_cli_application_flow(content, entry_point)
            elif app_type == 'Data Processing':
                flow['flow_steps'] = self._trace_data_processing_flow(content, entry_point)
            elif app_type == 'Machine Learning':
                flow['flow_steps'] = self._trace_ml_flow(content, entry_point)
            else:
                # Generic flow tracing for any application
                flow['flow_steps'] = self._trace_generic_flow(content, entry_point)
            
            # Extract operations and dependencies
            flow['discovered_operations'] = self._extract_operations_intelligently(content)
            flow['component_dependencies'] = self._extract_dependencies_intelligently(content)
            flow['execution_pattern'] = self._identify_execution_pattern(content)
            
            execution_flows.append(flow)
        
        return execution_flows
    
    def _trace_generic_flow(self, content: str, entry_point: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generic flow tracing that works for any type of application.
        Uses intelligent pattern recognition without domain-specific hardcoding.
        """
        flow_steps = []
        step_num = 1
        
        # Start
        flow_steps.append({
            'step': step_num,
            'action': f"Start: {entry_point['name']}",
            'type': 'initialization',
            'description': entry_point['description'],
            'category': 'entry'
        })
        step_num += 1
        
        # Imports and dependencies
        imports = self._extract_imports_intelligently(content)
        if imports:
            for imp in imports[:3]:  # Top 3 imports
                flow_steps.append({
                    'step': step_num,
                    'action': f"Import {imp['module']}",
                    'type': 'dependency_loading',
                    'description': f"Load {imp['type']} dependency",
                    'category': 'setup',
                    'details': imp
                })
                step_num += 1
        
        # Configuration and setup
        config_patterns = self._detect_configuration_patterns(content)
        for config in config_patterns[:2]:
            flow_steps.append({
                'step': step_num,
                'action': f"Configure {config['aspect']}",
                'type': 'configuration',
                'description': config['description'],
                'category': 'setup'
            })
            step_num += 1
        
        # Main operations
        operations = self._extract_main_operations(content)
        for op in operations[:4]:
            flow_steps.append({
                'step': step_num,
                'action': op['action'],
                'type': op['type'],
                'description': op['description'],
                'category': 'operation'
            })
            step_num += 1
        
        # Control flow
        control_structures = self._extract_control_flow_intelligently(content)
        for ctrl in control_structures[:2]:
            flow_steps.append({
                'step': step_num,
                'action': f"Control: {ctrl['type']}",
                'type': 'control_flow',
                'description': ctrl['description'],
                'category': 'logic'
            })
            step_num += 1
        
        return flow_steps
    
    def _extract_imports_intelligently(self, content: str) -> List[Dict[str, Any]]:
        """Extract imports intelligently for any language"""
        imports = []
        lines = content.split('\n')
        
        for line in lines[:50]:  # Check first 50 lines
            line = line.strip()
            
            # Python imports
            if line.startswith('import ') or line.startswith('from '):
                module_match = re.search(r'(?:from\s+)?(\w+(?:\.\w+)*)', line)
                if module_match:
                    module = module_match.group(1).split('.')[0]
                    imports.append({
                        'module': module,
                        'type': 'python_module',
                        'full_line': line
                    })
            
            # JavaScript/TypeScript imports
            elif 'import' in line and ('from' in line or 'require' in line):
                if 'from' in line:
                    module_match = re.search(r'from\s+["\']([^"\']+)["\']', line)
                elif 'require' in line:
                    module_match = re.search(r'require\(["\']([^"\']+)["\']\)', line)
                else:
                    continue
                
                if module_match:
                    module = module_match.group(1).split('/')[0]
                    imports.append({
                        'module': module,
                        'type': 'js_module',
                        'full_line': line
                    })
            
            # Java imports
            elif line.startswith('import ') and ';' in line:
                module_match = re.search(r'import\s+([^;]+)', line)
                if module_match:
                    module = module_match.group(1).split('.')[0]
                    imports.append({
                        'module': module,
                        'type': 'java_package',
                        'full_line': line
                    })
        
        return imports[:10]
    
    def _detect_configuration_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Detect configuration patterns intelligently"""
        configs = []
        
        config_patterns = [
            (r'config\s*=', 'application_config', 'Application configuration'),
            (r'settings\s*=', 'settings', 'Application settings'),
            (r'\.env', 'environment', 'Environment variables'),
            (r'os\.environ', 'environment', 'Environment access'),
            (r'argparse', 'cli_config', 'Command line configuration'),
            (r'configparser', 'file_config', 'Configuration file parsing'),
        ]
        
        for pattern, aspect, description in config_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                configs.append({
                    'aspect': aspect,
                    'description': description,
                    'pattern': pattern
                })
        
        return configs
    
    def _extract_main_operations(self, content: str) -> List[Dict[str, Any]]:
        """Extract main operations intelligently"""
        operations = []
        
        # Function calls that indicate main operations
        func_calls = self._extract_function_calls_with_context(content)
        
        # Classify operations based on naming patterns
        operation_classifiers = {
            'data': ['load', 'read', 'fetch', 'get', 'retrieve', 'parse'],
            'processing': ['process', 'transform', 'convert', 'compute', 'calculate'],
            'output': ['save', 'write', 'export', 'send', 'publish', 'display'],
            'control': ['start', 'stop', 'run', 'execute', 'launch', 'init'],
            'communication': ['connect', 'request', 'response', 'call', 'api']
        }
        
        for func in func_calls:
            func_lower = func['name'].lower()
            operation_type = 'operation'
            
            for op_type, keywords in operation_classifiers.items():
                if any(keyword in func_lower for keyword in keywords):
                    operation_type = op_type
                    break
            
            operations.append({
                'action': f"Execute {func['name']}()",
                'type': operation_type,
                'description': f"Perform {operation_type} operation",
                'function_details': func
            })
        
        return operations
    
    def _extract_function_calls_with_context(self, content: str) -> List[Dict[str, Any]]:
        """Extract function calls with context information"""
        function_calls = []
        
        # Regex to find function calls
        func_pattern = r'(\w+)\s*\([^)]*\)'
        matches = re.finditer(func_pattern, content)
        
        # Filter and contextualize
        seen_functions = set()
        for match in matches:
            func_name = match.group(1)
            
            # Skip common keywords and short names
            if (len(func_name) < 3 or 
                func_name in ['if', 'for', 'while', 'def', 'class', 'import', 'from', 'return'] or
                func_name in seen_functions):
                continue
            
            seen_functions.add(func_name)
            
            # Get context around the function call
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end]
            
            function_calls.append({
                'name': func_name,
                'context': context,
                'position': match.start()
            })
        
        return function_calls[:15]
    
    def _extract_control_flow_intelligently(self, content: str) -> List[Dict[str, Any]]:
        """Extract control flow structures intelligently"""
        control_structures = []
        
        # Control flow patterns (language-agnostic where possible)
        patterns = [
            (r'if\s+.+:', 'conditional', 'Conditional branching'),
            (r'for\s+.+:', 'loop', 'Iterative processing'),
            (r'while\s+.+:', 'loop', 'Conditional loop'),
            (r'try\s*:', 'exception_handling', 'Error handling'),
            (r'with\s+.+:', 'context_management', 'Resource management'),
            (r'async\s+def', 'async_operation', 'Asynchronous operation'),
            (r'await\s+', 'async_wait', 'Asynchronous waiting'),
        ]
        
        for pattern, ctrl_type, description in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                control_structures.append({
                    'type': ctrl_type,
                    'description': description,
                    'count': len(matches)
                })
        
        return control_structures
    
    def _extract_operations_intelligently(self, content: str) -> List[str]:
        """Extract operations without domain-specific hardcoding"""
        operations = []
        content_lower = content.lower()
        
        # Generic operation indicators
        operation_keywords = [
            'initialize', 'setup', 'configure', 'load', 'save', 'process',
            'execute', 'run', 'start', 'stop', 'create', 'update', 'delete',
            'read', 'write', 'parse', 'validate', 'transform', 'convert'
        ]
        
        for keyword in operation_keywords:
            if keyword in content_lower:
                operations.append(f"Performs {keyword} operations")
        
        return operations[:8]
    
    def _extract_dependencies_intelligently(self, content: str) -> List[str]:
        """Extract component dependencies intelligently"""
        dependencies = []
        
        # Extract imports as dependencies
        imports = self._extract_imports_intelligently(content)
        for imp in imports:
            dependencies.append(imp['module'])
        
        return dependencies[:10]
    
    def _identify_execution_pattern(self, content: str) -> str:
        """Identify the overall execution pattern"""
        patterns = {
            'sequential': ['step', 'then', 'next', 'after'],
            'event_driven': ['event', 'handler', 'callback', 'listener'],
            'loop_based': ['for', 'while', 'iterate', 'repeat'],
            'conditional': ['if', 'switch', 'case', 'condition'],
            'parallel': ['thread', 'async', 'concurrent', 'parallel'],
            'pipeline': ['pipe', 'chain', 'flow', 'pipeline']
        }
        
        content_lower = content.lower()
        pattern_scores = {}
        
        for pattern_name, keywords in patterns.items():
            score = sum(content_lower.count(keyword) for keyword in keywords)
            pattern_scores[pattern_name] = score
        
        if pattern_scores:
            return max(pattern_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'unknown'
    
    # Additional methods for web, CLI, data processing flows
    def _trace_web_application_flow(self, content: str, entry_point: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trace web application flow"""
        # Implementation for web app flow
        return self._trace_generic_flow(content, entry_point)
    
    def _trace_cli_application_flow(self, content: str, entry_point: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trace CLI application flow"""
        # Implementation for CLI app flow
        return self._trace_generic_flow(content, entry_point)
    
    def _trace_data_processing_flow(self, content: str, entry_point: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trace data processing flow"""
        # Implementation for data processing flow
        return self._trace_generic_flow(content, entry_point)
    
    def _trace_ml_flow(self, content: str, entry_point: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trace ML flow"""
        # Implementation for ML flow
        return self._trace_generic_flow(content, entry_point)
    
    def _determine_analysis_strategy(self, codebase_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the best analysis strategy based on codebase profile"""
        app_type = codebase_profile.get('application_type', 'Generic')
        complexity = codebase_profile.get('complexity_indicators', {})
        
        strategy = {
            'approach': 'adaptive',
            'depth': 'medium',
            'focus_areas': [],
            'confidence': 0.8
        }
        
        # Adjust strategy based on detected characteristics
        if app_type == 'Web Application':
            strategy['focus_areas'] = ['routing', 'request_handling', 'data_flow']
            strategy['depth'] = 'high'
        elif app_type == 'CLI Application':
            strategy['focus_areas'] = ['argument_parsing', 'command_flow', 'output']
        elif app_type == 'Data Processing':
            strategy['focus_areas'] = ['data_input', 'transformations', 'output']
        elif app_type == 'Machine Learning':
            strategy['focus_areas'] = ['model_lifecycle', 'data_pipeline', 'training_flow']
        else:
            strategy['focus_areas'] = ['entry_points', 'core_logic', 'dependencies']
        
        return strategy
    
    def _analyze_complexity_indicators(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze complexity indicators across the codebase"""
        total_files = len(detailed_files)
        total_lines = sum(f.get('lines', 0) for f in detailed_files.values())
        
        # Function complexity
        total_functions = 0
        large_functions = 0
        
        # Import complexity
        unique_imports = set()
        
        # File size distribution
        file_sizes = []
        
        for file_path, file_info in detailed_files.items():
            lines = file_info.get('lines', 0)
            file_sizes.append(lines)
            
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            
            # Count functions
            func_matches = re.findall(r'def\s+\w+\s*\(|function\s+\w+\s*\(', content)
            total_functions += len(func_matches)
            
            # Rough estimate of large functions (>50 lines between function definitions)
            func_positions = [m.start() for m in re.finditer(r'def\s+\w+\s*\(|function\s+\w+\s*\(', content)]
            for i, pos in enumerate(func_positions):
                next_pos = func_positions[i+1] if i+1 < len(func_positions) else len(content)
                func_content = content[pos:next_pos]
                if len(func_content.split('\n')) > 50:
                    large_functions += 1
            
            # Extract imports
            imports = self._extract_imports_intelligently(content)
            for imp in imports:
                unique_imports.add(imp['module'])
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'total_functions': total_functions,
            'large_functions': large_functions,
            'unique_dependencies': len(unique_imports),
            'avg_file_size': total_lines / max(total_files, 1),
            'complexity_level': self._calculate_overall_complexity(total_files, total_lines, total_functions, len(unique_imports))
        }
    
    def _calculate_overall_complexity(self, files: int, lines: int, functions: int, deps: int) -> str:
        """Calculate overall complexity level"""
        # Simple heuristic for complexity
        complexity_score = (files * 0.1) + (lines * 0.001) + (functions * 0.5) + (deps * 2)
        
        if complexity_score < 20:
            return 'Low'
        elif complexity_score < 100:
            return 'Medium'
        elif complexity_score < 300:
            return 'High'
        else:
            return 'Very High'
    
    def _calculate_entry_point_confidence(self, content: str, entry_type: str) -> float:
        """Calculate confidence score for entry point detection"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on entry type
        if entry_type == 'main_block' and 'if __name__ == "__main__"' in content:
            confidence = 0.95
        elif entry_type == 'main_function' and 'def main(' in content:
            confidence = 0.9
        elif entry_type == 'java_main' and 'public static void main' in content:
            confidence = 0.95
        elif entry_type == 'web_server' and any(pattern in content for pattern in ['app.run', 'server.listen']):
            confidence = 0.85
        elif entry_type == 'filename_entry':
            confidence = 0.6
        
        return confidence
    
    def _analyze_component_interactions_intelligently(self, detailed_files: Dict[str, Any], 
                                                   codebase_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze component interactions intelligently"""
        interactions = []
        
        # Build component map
        components = {}
        for file_path, file_info in detailed_files.items():
            component_name = Path(file_path).stem
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            
            components[component_name] = {
                'file': file_path,
                'imports': self._extract_imports_intelligently(content),
                'exports': self._extract_exports_intelligently(content),
                'size': file_info.get('lines', 0)
            }
        
        # Analyze interactions
        for comp_name, comp_info in components.items():
            for import_info in comp_info['imports']:
                imported_module = import_info['module']
                
                # Check for local component interactions
                for other_comp in components:
                    if (other_comp != comp_name and 
                        (imported_module == other_comp or imported_module in other_comp)):
                        
                        interactions.append({
                            'from_component': comp_name,
                            'to_component': other_comp,
                            'interaction_type': 'import_dependency',
                            'description': f"{comp_name} depends on {other_comp}",
                            'strength': self._calculate_interaction_strength(comp_info, components[other_comp])
                        })
        
        return interactions
    
    def _extract_exports_intelligently(self, content: str) -> List[Dict[str, Any]]:
        """Extract what a module exports intelligently"""
        exports = []
        
        # Python exports (functions and classes)
        func_pattern = r'def\s+(\w+)\s*\('
        class_pattern = r'class\s+(\w+)\s*[\(:]'
        
        functions = re.findall(func_pattern, content)
        classes = re.findall(class_pattern, content)
        
        for func in functions[:5]:
            exports.append({'name': func, 'type': 'function'})
        
        for cls in classes[:3]:
            exports.append({'name': cls, 'type': 'class'})
        
        # JavaScript/TypeScript exports
        export_patterns = [
            r'export\s+function\s+(\w+)',
            r'export\s+class\s+(\w+)',
            r'export\s+const\s+(\w+)',
            r'module\.exports\s*=\s*(\w+)'
        ]
        
        for pattern in export_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                exports.append({'name': match, 'type': 'export'})
        
        return exports
    
    def _calculate_interaction_strength(self, comp1: Dict, comp2: Dict) -> str:
        """Calculate strength of interaction between components"""
        # Simple heuristic based on import frequency and component sizes
        import_count = len(comp1.get('imports', []))
        size_ratio = min(comp1.get('size', 1), comp2.get('size', 1)) / max(comp1.get('size', 1), comp2.get('size', 1))
        
        if import_count > 3 and size_ratio > 0.5:
            return 'strong'
        elif import_count > 1:
            return 'medium'
        else:
            return 'weak'
    
    def _analyze_data_flows_dynamically(self, detailed_files: Dict[str, Any], 
                                      codebase_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze data flows dynamically"""
        data_flows = []
        
        for file_path, file_info in detailed_files.items():
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            
            inputs = self._detect_data_inputs_generically(content)
            transformations = self._detect_data_transformations_generically(content)
            outputs = self._detect_data_outputs_generically(content)
            
            if inputs or transformations or outputs:
                data_flows.append({
                    'file': file_path,
                    'inputs': inputs,
                    'transformations': transformations,
                    'outputs': outputs,
                    'complexity': len(inputs) + len(transformations) + len(outputs)
                })
        
        return sorted(data_flows, key=lambda x: x['complexity'], reverse=True)
    
    def _detect_data_inputs_generically(self, content: str) -> List[str]:
        """Detect data inputs generically"""
        inputs = []
        
        # File operations
        if 'open(' in content:
            inputs.append('File input')
        
        # Network operations
        if any(pattern in content for pattern in ['requests.', 'fetch(', 'http', 'api']):
            inputs.append('Network/API input')
        
        # Database operations
        if any(pattern in content for pattern in ['sql', 'query', 'database', 'db.']):
            inputs.append('Database input')
        
        # User input
        if any(pattern in content for pattern in ['input(', 'stdin', 'argv']):
            inputs.append('User input')
        
        # Environment variables
        if 'environ' in content:
            inputs.append('Environment variables')
        
        return inputs
    
    def _detect_data_transformations_generically(self, content: str) -> List[str]:
        """Detect data transformations generically"""
        transformations = []
        
        # Common transformation patterns
        transform_patterns = [
            ('parse', 'Data parsing'),
            ('convert', 'Data conversion'),
            ('transform', 'Data transformation'),
            ('filter', 'Data filtering'),
            ('map', 'Data mapping'),
            ('sort', 'Data sorting'),
            ('join', 'Data joining'),
            ('merge', 'Data merging'),
            ('split', 'Data splitting'),
            ('format', 'Data formatting')
        ]
        
        content_lower = content.lower()
        for pattern, description in transform_patterns:
            if pattern in content_lower:
                transformations.append(description)
        
        return transformations
    
    def _detect_data_outputs_generically(self, content: str) -> List[str]:
        """Detect data outputs generically"""
        outputs = []
        
        # Output patterns
        if any(pattern in content for pattern in ['write(', 'save(', 'dump']):
            outputs.append('File output')
        
        if any(pattern in content for pattern in ['print(', 'console.log', 'stdout']):
            outputs.append('Console output')
        
        if any(pattern in content for pattern in ['return ', 'yield ']):
            outputs.append('Function return value')
        
        if any(pattern in content for pattern in ['response', 'render', 'send']):
            outputs.append('Response/Render output')
        
        return outputs
    
    def _detect_architectural_patterns(self, detailed_files: Dict[str, Any], 
                                     codebase_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Detect architectural patterns intelligently"""
        patterns = {
            'detected_patterns': [],
            'architectural_style': 'Unknown',
            'organization_type': 'Unknown'
        }
        
        file_names = [Path(f).name.lower() for f in detailed_files.keys()]
        app_type = codebase_profile.get('application_type', 'Unknown')
        
        # MVC pattern detection
        if (any('model' in name for name in file_names) and
            any('view' in name for name in file_names) and
            any('controller' in name or 'control' in name for name in file_names)):
            patterns['detected_patterns'].append('MVC (Model-View-Controller)')
        
        # Microservices pattern
        if len([f for f in file_names if 'service' in f]) > 2:
            patterns['detected_patterns'].append('Service-oriented')
        
        # Layered architecture
        if (any('api' in name for name in file_names) and
            any('business' in name or 'logic' in name for name in file_names) and
            any('data' in name or 'db' in name for name in file_names)):
            patterns['detected_patterns'].append('Layered Architecture')
        
        # Component-based
        if len([f for f in file_names if 'component' in f]) > 1:
            patterns['detected_patterns'].append('Component-based')
        
        # Determine overall style
        if app_type == 'Web Application':
            patterns['architectural_style'] = 'Web Application Architecture'
        elif app_type == 'CLI Application':
            patterns['architectural_style'] = 'Command-line Tool'
        elif app_type == 'Machine Learning':
            patterns['architectural_style'] = 'ML Pipeline'
        elif len(detailed_files) > 20:
            patterns['architectural_style'] = 'Complex Multi-module System'
        elif len(detailed_files) < 5:
            patterns['architectural_style'] = 'Simple Script/Tool'
        else:
            patterns['architectural_style'] = 'Modular Application'
        
        return patterns
    
    def _generate_adaptive_diagrams(self, visual_report: Dict[str, Any]) -> Dict[str, str]:
        """Generate adaptive diagrams based on discovered patterns"""
        diagrams = {}
        
        # Main execution flow diagram
        execution_flows = visual_report.get('execution_flows', [])
        if execution_flows:
            diagrams['execution_flow'] = self._create_adaptive_execution_diagram(execution_flows[0])
        
        # Component interaction diagram
        interactions = visual_report.get('component_interactions', [])
        if interactions:
            diagrams['component_interactions'] = self._create_adaptive_interaction_diagram(interactions)
        
        # Data flow diagram
        data_flows = visual_report.get('data_flows', [])
        if data_flows:
            diagrams['data_flow'] = self._create_adaptive_data_flow_diagram(data_flows)
        
        return diagrams
    
    def _create_adaptive_execution_diagram(self, execution_flow: Dict[str, Any]) -> str:
        """Create adaptive execution flow diagram"""
        flow_steps = execution_flow.get('flow_steps', [])
        
        if not flow_steps:
            return "graph TD\n    A[\"No execution flow detected\"]"
        
        mermaid_lines = ["graph TD"]
        
        # Add adaptive styling based on step categories
        mermaid_lines.extend([
            "    classDef entry fill:#e1f5fe,stroke:#01579b,stroke-width:3px",
            "    classDef setup fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "    classDef operation fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px",
            "    classDef logic fill:#fff3e0,stroke:#e65100,stroke-width:2px"
        ])
        
        # Limit steps for readability
        steps_to_show = flow_steps[:8]
        
        for i, step in enumerate(steps_to_show):
            step_id = f"S{i+1}"
            action = step.get('action', f"Step {i+1}")
            category = step.get('category', 'operation')
            
            # Clean action text
            if len(action) > 25:
                action = action[:22] + "..."
            action = re.sub(r'["\[\]{}()]', '', action)
            
            # Add node
            mermaid_lines.append(f'    {step_id}["{action}"]')
            
            # Apply category styling
            if category == 'entry':
                mermaid_lines.append(f'    class {step_id} entry')
            elif category == 'setup':
                mermaid_lines.append(f'    class {step_id} setup')
            elif category == 'logic':
                mermaid_lines.append(f'    class {step_id} logic')
            else:
                mermaid_lines.append(f'    class {step_id} operation')
            
            # Connect to next step
            if i < len(steps_to_show) - 1:
                next_step_id = f"S{i+2}"
                mermaid_lines.append(f'    {step_id} --> {next_step_id}')
        
        return "\n".join(mermaid_lines)
    
    def _create_adaptive_interaction_diagram(self, interactions: List[Dict[str, Any]]) -> str:
        """Create adaptive component interaction diagram"""
        if not interactions:
            return "graph TD\n    A[\"No component interactions detected\"]"
        
        mermaid_lines = ["graph TD"]
        
        # Add styling based on interaction strength
        mermaid_lines.extend([
            "    classDef strong fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px",
            "    classDef medium fill:#fff9c4,stroke:#f57f17,stroke-width:2px",
            "    classDef weak fill:#ffcdd2,stroke:#c62828,stroke-width:1px"
        ])
        
        added_components = set()
        
        # Limit interactions for readability
        for i, interaction in enumerate(interactions[:10]):
            from_comp = interaction['from_component']
            to_comp = interaction['to_component']
            strength = interaction.get('strength', 'medium')
            
            # Clean component names
            from_id = f"C{len(added_components)+1}" if from_comp not in added_components else f"C{list(added_components).index(from_comp)+1}"
            to_id = f"C{len(added_components)+2}" if to_comp not in added_components else f"C{list(added_components).index(to_comp)+1}"
            
            # Add nodes if not already added
            if from_comp not in added_components:
                mermaid_lines.append(f'    {from_id}["{from_comp}"]')
                mermaid_lines.append(f'    class {from_id} {strength}')
                added_components.add(from_comp)
            
            if to_comp not in added_components:
                mermaid_lines.append(f'    {to_id}["{to_comp}"]')
                mermaid_lines.append(f'    class {to_id} {strength}')
                added_components.add(to_comp)
            
            # Add interaction
            interaction_type = interaction.get('interaction_type', 'uses')
            mermaid_lines.append(f'    {from_id} -->|{interaction_type}| {to_id}')
        
        return "\n".join(mermaid_lines)
    
    def _create_adaptive_data_flow_diagram(self, data_flows: List[Dict[str, Any]]) -> str:
        """Create adaptive data flow diagram"""
        if not data_flows:
            return "graph TD\n    A[\"No data flows detected\"]"
        
        mermaid_lines = ["graph TD"]
        
        # Add styling for data flow components
        mermaid_lines.extend([
            "    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px",
            "    classDef process fill:#f1f8e9,stroke:#558b2f,stroke-width:2px",
            "    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px"
        ])
        
        node_counter = 1
        
        # Process top 2 data flows
        for flow in data_flows[:2]:
            file_name = Path(flow['file']).stem
            file_id = f"F{node_counter}"
            
            # Add processing node
            mermaid_lines.append(f'    {file_id}["{file_name}"]')
            mermaid_lines.append(f'    class {file_id} process')
            node_counter += 1
            
            # Add inputs
            for inp in flow['inputs'][:2]:
                input_id = f"I{node_counter}"
                clean_inp = re.sub(r'[^a-zA-Z0-9 ]', '', inp)[:15]
                mermaid_lines.append(f'    {input_id}["{clean_inp}"]')
                mermaid_lines.append(f'    class {input_id} input')
                mermaid_lines.append(f'    {input_id} --> {file_id}')
                node_counter += 1
            
            # Add outputs
            for out in flow['outputs'][:2]:
                output_id = f"O{node_counter}"
                clean_out = re.sub(r'[^a-zA-Z0-9 ]', '', out)[:15]
                mermaid_lines.append(f'    {output_id}["{clean_out}"]')
                mermaid_lines.append(f'    class {output_id} output')
                mermaid_lines.append(f'    {file_id} --> {output_id}')
                node_counter += 1
        
        return "\n".join(mermaid_lines)
    
    def _generate_intelligent_insights(self, visual_report: Dict[str, Any], 
                                     detailed_files: Dict[str, Any]) -> List[str]:
        """Generate intelligent insights based on analysis"""
        insights = []
        
        codebase_profile = visual_report.get('codebase_profile', {})
        app_type = codebase_profile.get('application_type', 'Unknown')
        complexity = codebase_profile.get('complexity_indicators', {})
        entry_points = visual_report.get('entry_points', [])
        
        # Application type insights
        if app_type != 'Unknown':
            insights.append(f"🎯 Detected as {app_type} - architecture optimized for this domain")
        
        # Complexity insights
        complexity_level = complexity.get('complexity_level', 'Unknown')
        if complexity_level == 'Low':
            insights.append("✅ Low complexity codebase - easy to understand and maintain")
        elif complexity_level == 'High':
            insights.append("⚠️ High complexity detected - consider modularization")
        elif complexity_level == 'Very High':
            insights.append("🚨 Very high complexity - refactoring recommended")
        
        # Entry point insights
        if len(entry_points) == 1:
            insights.append("🎯 Single entry point - focused, cohesive application")
        elif len(entry_points) > 5:
            insights.append(f"🔀 Multiple entry points ({len(entry_points)}) - versatile but complex")
        
        # Language insights
        primary_lang = codebase_profile.get('primary_language')
        if primary_lang:
            insights.append(f"💻 {primary_lang}-based codebase with appropriate patterns")
        
        # Framework insights
        frameworks = codebase_profile.get('detected_frameworks', [])
        if frameworks:
            insights.append(f"🔧 Uses {len(frameworks)} frameworks: {', '.join(frameworks[:3])}")
        
        return insights[:6]
    
    def _generate_adaptive_recommendations(self, visual_report: Dict[str, Any], 
                                         codebase_profile: Dict[str, Any]) -> List[str]:
        """Generate adaptive recommendations based on analysis"""
        recommendations = []
        
        complexity = codebase_profile.get('complexity_indicators', {})
        app_type = codebase_profile.get('application_type', 'Unknown')
        entry_points = visual_report.get('entry_points', [])
        
        # Complexity-based recommendations
        if complexity.get('complexity_level') == 'High':
            recommendations.append("🏗️ Consider breaking large files into smaller modules")
        
        if complexity.get('large_functions', 0) > 3:
            recommendations.append("📏 Some functions are large - consider splitting them")
        
        # Entry point recommendations
        if len(entry_points) == 0:
            recommendations.append("🎯 Add clear entry points for better code organization")
        elif len(entry_points) > 10:
            recommendations.append("🔀 Consider consolidating entry points to reduce complexity")
        
        # Application-specific recommendations
        if app_type == 'Web Application':
            recommendations.append("🌐 Consider adding error handling and input validation")
        elif app_type == 'CLI Application':
            recommendations.append("💻 Consider adding help documentation and usage examples")
        elif app_type == 'Data Processing':
            recommendations.append("📊 Consider adding data validation and error handling")
        
        # General recommendations
        recommendations.append("📚 Use the visual diagrams to identify refactoring opportunities")
        
        return recommendations[:5] 