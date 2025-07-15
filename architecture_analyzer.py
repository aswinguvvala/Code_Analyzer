# architecture_analyzer.py - Analyze and visualize code architecture

import ast
import re
import os
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, deque
import json

class ArchitectureAnalyzer:
    """
    Analyze code architecture and generate visualizations of component relationships.
    
    Think of this as an architect studying a building to understand:
    - How rooms connect to each other (module dependencies)
    - Which are the main structural elements (central components)
    - Whether there are any design problems (circular dependencies)
    - How information flows through the system (call graphs)
    
    This analyzer creates multiple types of architectural views:
    1. Module Dependency Graph - shows which files import which other files
    2. Function Call Graph - shows which functions call which other functions
    3. Class Hierarchy - shows inheritance relationships
    4. Data Flow Analysis - tracks how data moves through the system
    """
    
    def __init__(self):
        self.dependency_graph = defaultdict(set)  # Module -> Set of dependencies
        self.reverse_dependency_graph = defaultdict(set)  # Module -> Set of dependents
        self.function_calls = defaultdict(set)  # Function -> Set of called functions
        self.class_hierarchy = defaultdict(set)  # Class -> Set of parent classes
        self.file_metrics = {}  # File -> Architectural metrics
    
    def analyze_architecture(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for architecture analysis.
        
        Like a comprehensive architectural survey, this examines the structure
        from multiple angles to provide a complete picture.
        """
        architecture_report = {
            'module_dependencies': {},
            'dependency_metrics': {},
            'circular_dependencies': [],
            'central_components': [],
            'architectural_patterns': [],
            'complexity_metrics': {},
            'recommendations': [],
            'visualizations': {
                'dependency_graph': None,
                'call_graph': None,
                'module_clusters': []
            }
        }
        
        # Step 1: Build the dependency graph
        self._build_dependency_graph(detailed_files)
        
        # Step 2: Analyze dependency metrics
        architecture_report['dependency_metrics'] = self._calculate_dependency_metrics()
        
        # Step 3: Detect circular dependencies
        architecture_report['circular_dependencies'] = self._detect_circular_dependencies()
        
        # Step 4: Identify central components
        architecture_report['central_components'] = self._identify_central_components()
        
        # Step 5: Detect architectural patterns
        architecture_report['architectural_patterns'] = self._detect_architectural_patterns(detailed_files)
        
        # Step 6: Calculate complexity metrics
        architecture_report['complexity_metrics'] = self._calculate_architectural_complexity()
        
        # Step 7: Generate visualization data
        architecture_report['visualizations'] = self._generate_visualization_data()
        
        # Step 8: Generate recommendations
        architecture_report['recommendations'] = self._generate_architectural_recommendations(architecture_report)
        
        # Step 9: Export dependency graph for external visualization
        architecture_report['module_dependencies'] = self._export_dependency_graph()
        
        return architecture_report
    
    def _build_dependency_graph(self, detailed_files: Dict[str, Any]):
        """
        Build a graph of how modules depend on each other.
        
        This is like mapping out which rooms in a building connect to which other rooms.
        We look at import statements to understand dependencies.
        """
        # First pass: collect all module names
        all_modules = set()
        for file_path in detailed_files.keys():
            if file_path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                module_name = self._file_path_to_module_name(file_path)
                all_modules.add(module_name)
        
        # Second pass: analyze imports and build dependency graph
        for file_path, file_info in detailed_files.items():
            if not file_path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                continue
            
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            if not content:
                continue
            
            current_module = self._file_path_to_module_name(file_path)
            dependencies = self._extract_dependencies(content, file_path, all_modules)
            
            # Build the dependency graph
            for dep in dependencies:
                self.dependency_graph[current_module].add(dep)
                self.reverse_dependency_graph[dep].add(current_module)
            
            # Calculate file-level metrics
            self.file_metrics[current_module] = self._calculate_file_metrics(content, file_path)
    
    def _file_path_to_module_name(self, file_path: str) -> str:
        """
        Convert a file path to a module name.
        
        Examples:
        - src/utils/helpers.py -> utils.helpers
        - components/Button.jsx -> components.Button
        """
        # Remove file extension
        module_path = file_path.rsplit('.', 1)[0]
        
        # Convert path separators to dots
        module_name = module_path.replace('/', '.').replace('\\', '.')
        
        # Remove common prefixes
        for prefix in ['src.', 'lib.', 'app.']:
            if module_name.startswith(prefix):
                module_name = module_name[len(prefix):]
                break
        
        return module_name
    
    def _extract_dependencies(self, content: str, file_path: str, all_modules: Set[str]) -> Set[str]:
        """
        Extract dependencies from a file's import statements.
        
        This is like identifying which other rooms you need to go through
        to get to different parts of the building.
        """
        dependencies = set()
        
        if file_path.endswith('.py'):
            dependencies.update(self._extract_python_dependencies(content, all_modules))
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            dependencies.update(self._extract_javascript_dependencies(content, all_modules))
        
        return dependencies
    
    def _extract_python_dependencies(self, content: str, all_modules: Set[str]) -> Set[str]:
        """
        Extract Python import dependencies.
        
        Python has several import formats:
        - import module
        - from module import something
        - import module.submodule
        """
        dependencies = set()
        
        try:
            # Parse the Python code into an AST
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        # Check if this is a local module (in our codebase)
                        if self._is_local_module(module_name, all_modules):
                            dependencies.add(module_name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module
                        if self._is_local_module(module_name, all_modules):
                            dependencies.add(module_name)
        
        except SyntaxError:
            # If parsing fails, fall back to regex parsing
            dependencies.update(self._extract_python_dependencies_regex(content, all_modules))
        
        return dependencies
    
    def _extract_python_dependencies_regex(self, content: str, all_modules: Set[str]) -> Set[str]:
        """
        Fallback regex-based Python dependency extraction.
        """
        dependencies = set()
        
        # Match import statements
        import_patterns = [
            r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import'
        ]
        
        for line in content.split('\n'):
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module_name = match.group(1)
                    if self._is_local_module(module_name, all_modules):
                        dependencies.add(module_name)
        
        return dependencies
    
    def _extract_javascript_dependencies(self, content: str, all_modules: Set[str]) -> Set[str]:
        """
        Extract JavaScript/TypeScript import dependencies.
        
        JavaScript has several import formats:
        - import something from './module'
        - import { something } from './module'
        - const something = require('./module')
        """
        dependencies = set()
        
        # Regex patterns for different import styles
        import_patterns = [
            r'import\s+.*?\s+from\s+["\']([^"\']+)["\']',  # ES6 imports
            r'import\s+["\']([^"\']+)["\']',  # Side-effect imports
            r'require\s*\(\s*["\']([^"\']+)["\']\s*\)',  # CommonJS requires
        ]
        
        for line in content.split('\n'):
            for pattern in import_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    # Convert relative imports to module names
                    module_name = self._resolve_javascript_import(match)
                    if module_name and self._is_local_module(module_name, all_modules):
                        dependencies.add(module_name)
        
        return dependencies
    
    def _resolve_javascript_import(self, import_path: str) -> Optional[str]:
        """
        Resolve JavaScript import paths to module names.
        
        Examples:
        - './utils' -> utils
        - '../components/Button' -> components.Button
        - './helpers/index' -> helpers
        """
        if not import_path.startswith('.'):
            # External module, not a local dependency
            return None
        
        # Remove leading './' and '../'
        cleaned_path = import_path
        while cleaned_path.startswith('./') or cleaned_path.startswith('../'):
            if cleaned_path.startswith('./'):
                cleaned_path = cleaned_path[2:]
            else:
                cleaned_path = cleaned_path[3:]
        
        # Convert to module name format
        module_name = cleaned_path.replace('/', '.')
        
        # Remove common suffixes
        if module_name.endswith('.index'):
            module_name = module_name[:-6]
        
        return module_name if module_name else None
    
    def _is_local_module(self, module_name: str, all_modules: Set[str]) -> bool:
        """
        Determine if a module is local to the codebase (vs external library).
        """
        # Check exact match
        if module_name in all_modules:
            return True
        
        # Check if it's a submodule of any known module
        for known_module in all_modules:
            if module_name.startswith(known_module + '.') or known_module.startswith(module_name + '.'):
                return True
        
        return False
    
    def _calculate_file_metrics(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Calculate architectural metrics for a single file.
        
        These metrics help us understand the role and complexity of each component.
        """
        metrics = {
            'lines_of_code': len([line for line in content.split('\n') if line.strip()]),
            'function_count': 0,
            'class_count': 0,
            'complexity_score': 0,
            'abstraction_level': 'concrete'  # concrete, abstract, interface
        }
        
        if file_path.endswith('.py'):
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        metrics['function_count'] += 1
                    elif isinstance(node, ast.ClassDef):
                        metrics['class_count'] += 1
                        
                        # Check if it's an abstract class
                        for base in node.bases:
                            if isinstance(base, ast.Name) and 'Abstract' in base.id:
                                metrics['abstraction_level'] = 'abstract'
            
            except SyntaxError:
                pass
        
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            # Simple heuristics for JavaScript/TypeScript
            metrics['function_count'] = len(re.findall(r'\bfunction\b|\b=>\b', content))
            metrics['class_count'] = len(re.findall(r'\bclass\b', content))
            
            if 'interface' in content.lower() or 'abstract' in content.lower():
                metrics['abstraction_level'] = 'abstract'
        
        # Calculate complexity score based on various factors
        metrics['complexity_score'] = (
            metrics['lines_of_code'] * 0.1 +
            metrics['function_count'] * 2 +
            metrics['class_count'] * 3
        )
        
        return metrics
    
    def _calculate_dependency_metrics(self) -> Dict[str, Any]:
        """
        Calculate overall dependency metrics for the codebase.
        
        These metrics give us insights into the architectural health:
        - Fan-in: How many modules depend on this one (popularity)
        - Fan-out: How many modules this one depends on (coupling)
        - Instability: Ratio of fan-out to total dependencies
        """
        metrics = {
            'total_modules': len(self.dependency_graph),
            'total_dependencies': sum(len(deps) for deps in self.dependency_graph.values()),
            'average_fan_out': 0,
            'average_fan_in': 0,
            'most_coupled_modules': [],
            'most_stable_modules': [],
            'dependency_distribution': {}
        }
        
        if metrics['total_modules'] > 0:
            metrics['average_fan_out'] = metrics['total_dependencies'] / metrics['total_modules']
            
            # Calculate metrics for each module
            module_metrics = {}
            for module in self.dependency_graph.keys():
                fan_out = len(self.dependency_graph[module])
                fan_in = len(self.reverse_dependency_graph[module])
                total_coupling = fan_in + fan_out
                
                # Instability metric: I = fan_out / (fan_in + fan_out)
                # 0 = stable (many dependents, few dependencies)
                # 1 = unstable (few dependents, many dependencies)
                instability = fan_out / max(total_coupling, 1)
                
                module_metrics[module] = {
                    'fan_in': fan_in,
                    'fan_out': fan_out,
                    'instability': instability,
                    'total_coupling': total_coupling
                }
            
            # Find most coupled modules (high fan-in + fan-out)
            metrics['most_coupled_modules'] = sorted(
                module_metrics.items(),
                key=lambda x: x[1]['total_coupling'],
                reverse=True
            )[:5]
            
            # Find most stable modules (low instability, high fan-in)
            metrics['most_stable_modules'] = sorted(
                [(module, data) for module, data in module_metrics.items() if data['fan_in'] > 0],
                key=lambda x: x[1]['instability']
            )[:5]
            
            metrics['average_fan_in'] = sum(data['fan_in'] for data in module_metrics.values()) / len(module_metrics)
        
        return metrics
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the module graph.
        
        Circular dependencies are like having a loop in your building's hallways -
        they create confusion and make the system harder to understand and maintain.
        
        We use a depth-first search to find strongly connected components.
        """
        def dfs_visit(node: str, visited: Set[str], rec_stack: Set[str], path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph[node]:
                if neighbor not in visited:
                    cycle = dfs_visit(neighbor, visited, rec_stack, path)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found a cycle - extract the cycle from the path
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            path.pop()
            return None
        
        cycles = []
        visited = set()
        
        for node in self.dependency_graph:
            if node not in visited:
                cycle = dfs_visit(node, visited, set(), [])
                if cycle and len(cycle) > 2:  # Only report cycles with more than 2 nodes
                    cycles.append(cycle)
        
        return cycles
    
    def _identify_central_components(self) -> List[Dict[str, Any]]:
        """
        Identify the most central/important components in the architecture.
        
        Central components are like the main pillars of a building - they support
        many other parts and are critical to the overall structure.
        
        We use several centrality measures:
        - Degree centrality: How many connections a node has
        - Betweenness centrality: How often a node lies on paths between other nodes
        """
        central_components = []
        
        # Calculate degree centrality (simple but effective)
        for module in self.dependency_graph:
            degree = len(self.dependency_graph[module]) + len(self.reverse_dependency_graph[module])
            if degree > 0:
                central_components.append({
                    'module': module,
                    'degree_centrality': degree,
                    'fan_in': len(self.reverse_dependency_graph[module]),
                    'fan_out': len(self.dependency_graph[module]),
                    'centrality_type': self._classify_centrality_type(
                        len(self.reverse_dependency_graph[module]),
                        len(self.dependency_graph[module])
                    )
                })
        
        # Sort by degree centrality
        central_components.sort(key=lambda x: x['degree_centrality'], reverse=True)
        
        return central_components[:10]  # Return top 10
    
    def _classify_centrality_type(self, fan_in: int, fan_out: int) -> str:
        """
        Classify the type of centrality based on fan-in and fan-out patterns.
        """
        if fan_in > fan_out * 2:
            return 'hub'  # Many modules depend on this one
        elif fan_out > fan_in * 2:
            return 'broker'  # This module depends on many others
        elif fan_in > 3 and fan_out > 3:
            return 'bridge'  # Connects many different parts
        else:
            return 'connector'  # General connector
    
    def _detect_architectural_patterns(self, detailed_files: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect common architectural patterns in the codebase.
        
        This is like recognizing common building styles or design patterns.
        We look for evidence of well-known patterns like MVC, layered architecture, etc.
        """
        patterns = []
        
        # Look for MVC pattern
        mvc_evidence = self._detect_mvc_pattern()
        if mvc_evidence:
            patterns.append(mvc_evidence)
        
        # Look for layered architecture
        layered_evidence = self._detect_layered_architecture()
        if layered_evidence:
            patterns.append(layered_evidence)
        
        # Look for microservices patterns
        microservices_evidence = self._detect_microservices_pattern(detailed_files)
        if microservices_evidence:
            patterns.append(microservices_evidence)
        
        return patterns
    
    def _detect_mvc_pattern(self) -> Optional[Dict[str, Any]]:
        """
        Detect Model-View-Controller pattern.
        """
        models = [m for m in self.dependency_graph.keys() if 'model' in m.lower()]
        views = [m for m in self.dependency_graph.keys() if 'view' in m.lower() or 'template' in m.lower()]
        controllers = [m for m in self.dependency_graph.keys() if 'controller' in m.lower() or 'handler' in m.lower()]
        
        if models and views and controllers:
            return {
                'pattern': 'MVC (Model-View-Controller)',
                'confidence': 0.8,
                'evidence': {
                    'models': len(models),
                    'views': len(views),
                    'controllers': len(controllers)
                },
                'description': 'Separation of concerns with distinct model, view, and controller components'
            }
        
        return None
    
    def _detect_layered_architecture(self) -> Optional[Dict[str, Any]]:
        """
        Detect layered architecture pattern.
        """
        layers = {
            'data': [m for m in self.dependency_graph.keys() if any(layer in m.lower() for layer in ['data', 'dao', 'repository'])],
            'service': [m for m in self.dependency_graph.keys() if 'service' in m.lower()],
            'controller': [m for m in self.dependency_graph.keys() if any(layer in m.lower() for layer in ['controller', 'api', 'handler'])],
            'presentation': [m for m in self.dependency_graph.keys() if any(layer in m.lower() for layer in ['view', 'ui', 'component'])]
        }
        
        layer_count = sum(1 for layer_modules in layers.values() if layer_modules)
        
        if layer_count >= 3:
            return {
                'pattern': 'Layered Architecture',
                'confidence': 0.7,
                'evidence': {layer: len(modules) for layer, modules in layers.items() if modules},
                'description': 'Application organized in distinct layers with clear separation of concerns'
            }
        
        return None
    
    def _detect_microservices_pattern(self, detailed_files: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect microservices architecture patterns.
        """
        # Look for common microservices indicators
        service_indicators = []
        
        # Check for Docker files
        if any('dockerfile' in f.lower() for f in detailed_files.keys()):
            service_indicators.append('containerization')
        
        # Check for API endpoints
        api_files = [f for f in detailed_files.keys() if any(indicator in f.lower() for indicator in ['api', 'endpoint', 'route'])]
        if api_files:
            service_indicators.append('api_endpoints')
        
        # Check for service discovery patterns
        if any('service' in m.lower() and 'discovery' in m.lower() for m in self.dependency_graph.keys()):
            service_indicators.append('service_discovery')
        
        if len(service_indicators) >= 2:
            return {
                'pattern': 'Microservices Architecture',
                'confidence': 0.6,
                'evidence': service_indicators,
                'description': 'Application appears to follow microservices patterns with service separation'
            }
        
        return None
    
    def _calculate_architectural_complexity(self) -> Dict[str, Any]:
        """
        Calculate overall architectural complexity metrics.
        
        Complexity affects maintainability, testability, and understanding.
        """
        total_modules = len(self.dependency_graph)
        total_edges = sum(len(deps) for deps in self.dependency_graph.values())
        
        # Calculate graph density: actual edges / possible edges
        max_possible_edges = total_modules * (total_modules - 1)
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Calculate average path length (simplified estimation)
        avg_path_length = self._estimate_average_path_length()
        
        # Calculate modularity (how well modules cluster together)
        modularity = self._calculate_modularity()
        
        return {
            'total_modules': total_modules,
            'total_dependencies': total_edges,
            'graph_density': round(density, 3),
            'average_path_length': round(avg_path_length, 2),
            'modularity': round(modularity, 3),
            'complexity_level': self._classify_complexity_level(density, total_modules)
        }
    
    def _estimate_average_path_length(self) -> float:
        """
        Estimate the average path length in the dependency graph.
        """
        if not self.dependency_graph:
            return 0
        
        # Simple estimation based on graph structure
        total_modules = len(self.dependency_graph)
        total_edges = sum(len(deps) for deps in self.dependency_graph.values())
        
        if total_edges == 0:
            return float('inf')
        
        # Rough approximation: log(n) for well-connected graphs
        import math
        return max(1, math.log(total_modules) if total_modules > 1 else 1)
    
    def _calculate_modularity(self) -> float:
        """
        Calculate a simple modularity score based on naming patterns.
        """
        # Group modules by common prefixes
        groups = defaultdict(list)
        for module in self.dependency_graph.keys():
            prefix = module.split('.')[0] if '.' in module else module
            groups[prefix].append(module)
        
        # Calculate intra-group vs inter-group connections
        intra_group_edges = 0
        total_edges = 0
        
        for group_modules in groups.values():
            for module in group_modules:
                for dep in self.dependency_graph[module]:
                    total_edges += 1
                    if dep in group_modules:
                        intra_group_edges += 1
        
        return intra_group_edges / max(total_edges, 1)
    
    def _classify_complexity_level(self, density: float, total_modules: int) -> str:
        """
        Classify the overall complexity level of the architecture.
        """
        if total_modules < 10:
            return 'simple'
        elif total_modules < 50 and density < 0.1:
            return 'moderate'
        elif total_modules < 100 and density < 0.2:
            return 'complex'
        else:
            return 'very_complex'
    
    def _generate_visualization_data(self) -> Dict[str, Any]:
        """
        Generate data structures for creating architectural visualizations.
        
        This prepares the data in formats that can be easily consumed
        by visualization libraries like D3.js, Plotly, or Graphviz.
        """
        nodes = []
        edges = []
        
        # Create nodes
        for module in self.dependency_graph.keys():
            metrics = self.file_metrics.get(module, {})
            node = {
                'id': module,
                'label': module.split('.')[-1],  # Use last part as label
                'size': metrics.get('complexity_score', 1),
                'group': module.split('.')[0] if '.' in module else 'root',
                'metrics': metrics
            }
            nodes.append(node)
        
        # Create edges
        for source, targets in self.dependency_graph.items():
            for target in targets:
                edges.append({
                    'source': source,
                    'target': target,
                    'weight': 1
                })
        
        return {
            'dependency_graph': {
                'nodes': nodes,
                'edges': edges
            },
            'format': 'networkx_compatible',
            'statistics': {
                'node_count': len(nodes),
                'edge_count': len(edges)
            }
        }
    
    def _generate_architectural_recommendations(self, architecture_report: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations for improving the architecture.
        """
        recommendations = []
        
        # Check for circular dependencies
        circular_deps = architecture_report.get('circular_dependencies', [])
        if circular_deps:
            recommendations.append(f"🔄 Fix {len(circular_deps)} circular dependencies to improve maintainability")
        
        # Check for high coupling
        metrics = architecture_report.get('dependency_metrics', {})
        if metrics.get('average_fan_out', 0) > 5:
            recommendations.append("🔗 Consider reducing coupling - some modules depend on too many others")
        
        # Check for architectural complexity
        complexity = architecture_report.get('complexity_metrics', {})
        complexity_level = complexity.get('complexity_level', 'simple')
        
        if complexity_level == 'very_complex':
            recommendations.append("🏗️ Architecture is very complex - consider refactoring into smaller, focused modules")
        elif complexity_level == 'complex':
            recommendations.append("📐 Architecture is getting complex - monitor growth and consider modularization")
        
        # Check for central components
        central_components = architecture_report.get('central_components', [])
        if central_components:
            most_central = central_components[0]
            if most_central['degree_centrality'] > 10:
                recommendations.append(f"⚡ '{most_central['module']}' is highly central - ensure it's well-tested and stable")
        
        # Check for patterns
        patterns = architecture_report.get('architectural_patterns', [])
        if not patterns:
            recommendations.append("🎯 Consider adopting architectural patterns (MVC, layers) for better organization")
        
        if not recommendations:
            recommendations.append("✅ Architecture looks well-organized!")
        
        return recommendations
    
    def _export_dependency_graph(self) -> Dict[str, Any]:
        """
        Export the dependency graph in a format suitable for external analysis.
        """
        return {
            'dependencies': {module: list(deps) for module, deps in self.dependency_graph.items()},
            'reverse_dependencies': {module: list(deps) for module, deps in self.reverse_dependency_graph.items()},
            'file_metrics': self.file_metrics
        }