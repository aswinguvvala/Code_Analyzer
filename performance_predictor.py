# performance_predictor.py
import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PerformanceIssue:
    """Represents a detected performance issue."""
    severity: str  # 'low', 'medium', 'high', 'critical'
    type: str      # 'algorithm', 'memory', 'io', 'database', etc.
    location: str  # File and line number
    description: str
    impact: str    # Expected performance impact
    suggestion: str
    code_snippet: Optional[str] = None
    estimated_complexity: Optional[str] = None

class PerformancePredictor:
    """
    Predicts performance bottlenecks by analyzing code patterns.
    This is like having a performance expert review your code before it runs.
    """
    
    def __init__(self):
        self.issues = []
        self.performance_score = 100  # Start with perfect score
        self.analysis_stats = {
            'files_analyzed': 0,
            'functions_analyzed': 0,
            'issues_found': defaultdict(int)
        }
        self.project_context = {}
        self.performance_standards = {}

    def analyze_performance(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for performance analysis.
        Analyzes the entire codebase for potential performance issues.
        """
        print("🚀 Starting performance prediction analysis...")
        
        # Reset for new analysis
        self.issues = []
        self.performance_score = 100
        
        # Analyze each file
        for file_path, file_info in detailed_files.items():
            if file_info.get('extension') in ['.py', '.js', '.ts', '.java']:
                self._analyze_file_performance(file_path, file_info)
                self.analysis_stats['files_analyzed'] += 1
        
        # Analyze cross-file patterns
        self._analyze_architectural_performance(detailed_files)
        
        # Calculate final score
        self._calculate_performance_score()
        
        # Generate report
        return self._generate_performance_report()

    def _analyze_file_performance(self, file_path: str, file_info: Dict[str, Any]):
        """Analyze a single file for performance issues."""
        content = file_info.get('full_content', file_info.get('content_preview', ''))
        
        if file_info.get('extension') == '.py':
            self._analyze_python_performance(file_path, content)
        elif file_info.get('extension') in ['.js', '.ts']:
            self._analyze_javascript_performance(file_path, content)
        
        # Language-agnostic analyses
        self._analyze_general_patterns(file_path, content, file_info)

    def _analyze_python_performance(self, file_path: str, content: str):
        """
        Python-specific performance analysis.
        We look for common Python performance anti-patterns.
        """
        try:
            tree = ast.parse(content)
            
            # Walk the AST looking for performance issues
            for node in ast.walk(tree):
                self._check_nested_loops(node, file_path, content)
                self._check_inefficient_operations(node, file_path, content)
                self._check_memory_issues(node, file_path, content)
                self._check_database_patterns(node, file_path, content)
                
                if isinstance(node, ast.FunctionDef):
                    self.analysis_stats['functions_analyzed'] += 1
                    self._analyze_function_performance(node, file_path, content)
                    
        except (SyntaxError, ValueError):
            # Skip files with syntax errors or other parsing issues
            pass
    
    def _check_nested_loops(self, node: ast.AST, file_path: str, content: str):
        """
        Detect nested loops that could cause O(n²) or worse complexity.
        This is one of the most common performance killers.
        """
        if isinstance(node, (ast.For, ast.While)):
            # Check if this loop contains another loop
            nested_loops = []
            for child in ast.walk(node):
                if child != node and isinstance(child, (ast.For, ast.While)):
                    nested_loops.append(child)
            
            if nested_loops:
                # Calculate depth and estimate complexity
                depth = self._calculate_loop_depth(node)
                
                if depth >= 2:
                    complexity = f"O(n^{depth})"
                    severity = 'medium' if depth == 2 else 'high' if depth == 3 else 'critical'
                    
                    # Extract the problematic code
                    lines = content.splitlines()
                    start_line = node.lineno - 1
                    end_line = min(start_line + 10, len(lines))  # Show first 10 lines
                    code_snippet = '\n'.join(lines[start_line:end_line])
                    
                    # Check if it's operating on large data
                    impact = self._estimate_loop_impact(node, content)
                    
                    self.issues.append(PerformanceIssue(
                        severity=severity,
                        type='algorithm',
                        location=f"{file_path}:{node.lineno}",
                        description=f"Nested loops detected with complexity {complexity}",
                        impact=impact,
                        suggestion=self._suggest_loop_optimization(node, depth),
                        code_snippet=code_snippet,
                        estimated_complexity=complexity
                    ))
                    
                    self.analysis_stats['issues_found']['nested_loops'] += 1
    
    def _calculate_loop_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate the maximum nesting depth of loops."""
        max_depth = current_depth
        
        if isinstance(node, (ast.For, ast.While)):
            current_depth += 1
            max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            child_depth = self._calculate_loop_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _estimate_loop_impact(self, loop_node: ast.AST, content: str) -> str:
        """
        Estimate the real-world impact of a loop based on what it's iterating over.
        """
        # Look for common patterns that indicate data size
        content_lower = content.lower()
        
        # Check what the loop is iterating over
        if isinstance(loop_node, ast.For):
            iter_target = ast.unparse(loop_node.iter) if hasattr(ast, 'unparse') else str(loop_node.iter)
            
            if 'database' in iter_target or 'query' in iter_target or '.all()' in iter_target:
                return "🚨 HIGH: Iterating over database results - could process thousands of records"
            elif 'file' in iter_target or 'readlines' in iter_target:
                return "⚠️ MEDIUM: Processing file contents - impact depends on file size"
            elif 'range' in iter_target:
                # Try to extract the range limit
                if 'range(len(' in iter_target:
                    return "⚠️ MEDIUM: Impact depends on data structure size"
                else:
                    return "ℹ️ LOW: Fixed range iteration"
        
        # Check surrounding context
        if 'user' in content_lower or 'customer' in content_lower:
            return "⚠️ MEDIUM: Processing user data - could scale with user growth"
        elif 'api' in content_lower or 'request' in content_lower:
            return "🚨 HIGH: Could impact API response times"
        
        return "ℹ️ UNKNOWN: Impact depends on data size"
    
    def _suggest_loop_optimization(self, node: ast.AST, depth: int) -> str:
        """Generate specific optimization suggestions for nested loops."""
        suggestions = []
        
        if depth == 2:
            suggestions.append("Consider using a hash map/dictionary for O(1) lookups instead of inner loop")
            suggestions.append("Use set operations if checking membership")
            suggestions.append("Pre-sort data and use binary search if applicable")
        elif depth >= 3:
            suggestions.append("🚨 CRITICAL: Refactor algorithm - consider dynamic programming or divide-and-conquer")
            suggestions.append("Use specialized data structures (e.g., KD-tree for spatial data)")
            suggestions.append("Consider preprocessing data or caching intermediate results")
        
        # Check if it's a search operation
        if isinstance(node, ast.For) and any(isinstance(n, ast.If) for n in ast.walk(node)):
            suggestions.append("If searching, consider indexing data structure or using bisect")
        
        return " | ".join(suggestions)
    
    def _check_inefficient_operations(self, node: ast.AST, file_path: str, content: str):
        """
        Check for inefficient operations like string concatenation in loops,
        list operations that could be vectorized, etc.
        """
        # String concatenation in loops
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                    # Check if it's string concatenation
                    if isinstance(child.target, ast.Name):
                        var_name = child.target.id
                        # Simple heuristic - if variable name suggests string
                        if any(s in var_name.lower() for s in ['str', 'text', 'result', 'output']):
                            self.issues.append(PerformanceIssue(
                                severity='medium',
                                type='memory',
                                location=f"{file_path}:{child.lineno}",
                                description="String concatenation in loop detected",
                                impact="Creates new string object each iteration - O(n²) for n concatenations",
                                suggestion="Use `list.append()` and `''.join()` instead, or `io.StringIO` for large strings"
                            ))
                            self.analysis_stats['issues_found']['string_concat'] += 1
        
        # List comprehension opportunities
        if isinstance(node, ast.For):
            # Check if loop is just appending to a list
            loop_body = node.body
            if (len(loop_body) == 1 and 
                isinstance(loop_body[0], ast.Expr) and
                isinstance(loop_body[0].value, ast.Call) and
                isinstance(loop_body[0].value.func, ast.Attribute) and
                loop_body[0].value.func.attr == 'append'):
                
                # Further check to avoid false positives with conditional appends
                if not any(isinstance(n, ast.If) for n in ast.walk(node)):
                    self.issues.append(PerformanceIssue(
                        severity='low',
                        type='optimization',
                        location=f"{file_path}:{node.lineno}",
                        description="Loop could be replaced with list comprehension",
                        impact="List comprehensions are often faster and more readable",
                        suggestion="Replace with: `[expression for item in iterable]`"
                    ))
                    self.analysis_stats['issues_found']['comprehension_opportunity'] += 1
    
    def _check_memory_issues(self, node: ast.AST, file_path: str, content: str):
        """
        Check for potential memory issues like loading large files entirely,
        creating unnecessary copies, memory leaks, etc.
        """
        # Check for reading entire files
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and 
                node.func.attr in ['read', 'readlines']):
                
                # Check if it's a file operation
                context_line = content.splitlines()[node.lineno - 1] if node.lineno <= len(content.splitlines()) else ""
                if 'open(' in context_line or 'file' in context_line.lower():
                    self.issues.append(PerformanceIssue(
                        severity='medium',
                        type='memory',
                        location=f"{file_path}:{node.lineno}",
                        description="Reading entire file into memory",
                        impact="Could cause memory issues with large files (GB+)",
                        suggestion="Use generators or process file line by line: for line in file:"
                    ))
                    self.analysis_stats['issues_found']['memory_file_read'] += 1
            
            # Check for unnecessary list creation
            elif (isinstance(node.func, ast.Name) and 
                  node.func.id == 'list' and 
                  node.args):
                # Check if converting a generator unnecessarily
                self.issues.append(PerformanceIssue(
                    severity='low',
                    type='memory',
                    location=f"{file_path}:{node.lineno}",
                    description="Unnecessary list creation from iterator",
                    impact="Creates full list in memory when iterator might suffice",
                    suggestion="Consider using the iterator directly if you don't need random access"
                ))
                self.analysis_stats['issues_found']['unnecessary_list'] += 1

        # Simple check for reading large files into memory
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in ['read', 'readlines']:
                # Check if it's from a file object
                if isinstance(node.func.value, ast.Name):
                    self.issues.append(PerformanceIssue(
                        severity='high',
                        type='memory',
                        location=f"{file_path}:{node.lineno}",
                        description="Potential large file read into memory",
                        impact="Reading large files at once can cause high memory usage",
                        suggestion="Process file line-by-line or in chunks, e.g., `for line in file:`"
                    ))
                    self.analysis_stats['issues_found']['large_file_read'] += 1
    
    def _check_database_patterns(self, node: ast.AST, file_path: str, content: str):
        """
        Check for database-related performance issues like N+1 queries,
        missing indexes, inefficient queries, etc.
        """
        # N+1 query pattern detection
        if isinstance(node, ast.For):
            # Look for database queries inside loops
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    call_str = ast.unparse(child) if hasattr(ast, 'unparse') else str(child)
                    
                    # Common ORM patterns
                    if any(pattern in call_str for pattern in [
                        '.get(', '.filter(', '.query', '.find(', 'select(',
                        'SELECT', 'findOne', 'findAll'
                    ]):
                        self.issues.append(PerformanceIssue(
                            severity='high',
                            type='database',
                            location=f"{file_path}:{child.lineno}",
                            description="Database query inside loop (N+1 problem)",
                            impact="Each iteration makes a database call - catastrophic for large datasets",
                            suggestion="Use eager loading (e.g., select_related, prefetch_related) or batch queries"
                        ))
                        self.analysis_stats['issues_found']['n_plus_one'] += 1
                        break
        
        # Extremely basic check for N+1 query pattern
        # A real implementation would need to track ORM usage more deeply
        contains_loop = True
        contains_db_call = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and 'query' in str(ast.dump(child)).lower():
                contains_db_call = True
        
        if contains_loop and contains_db_call:
            self.issues.append(PerformanceIssue(
                severity='critical',
                type='database',
                location=f"{file_path}:{node.lineno}",
                description="Potential N+1 database query in a loop",
                impact="Can lead to thousands of unnecessary database calls",
                suggestion="Use eager loading (e.g., `select_related`, `prefetch_related` in Django) or batch queries"
            ))
            self.analysis_stats['issues_found']['n_plus_1_query'] += 1

    def _analyze_function_performance(self, func_node: ast.FunctionDef, file_path: str, content: str):
        """
        Analyze individual function performance characteristics.
        """
        # 1. Unbounded recursion
        if self._is_recursive_function(func_node) and not self._has_memoization(func_node, content):
            self.issues.append(PerformanceIssue(
                severity='high',
                type='algorithm',
                location=f"{file_path}:{func_node.lineno}",
                description=f"Recursive function '{func_node.name}' without memoization",
                impact="Risk of stack overflow and re-computation of results for same inputs",
                suggestion="Use `@functools.lru_cache` or a manual caching mechanism"
            ))
            self.analysis_stats['issues_found']['unbounded_recursion'] += 1
            
        # 2. High complexity
        complexity = self._calculate_complexity(func_node)
        if complexity > 15:
            self.issues.append(PerformanceIssue(
                severity='medium',
                type='maintainability',
                location=f"{file_path}:{func_node.lineno}",
                description=f"High cyclomatic complexity in function '{func_node.name}' ({complexity})",
                impact="Difficult to test, understand, and maintain. Higher chance of bugs.",
                suggestion="Refactor into smaller, more focused functions"
            ))
            self.analysis_stats['issues_found']['high_complexity'] += 1
    
    def _is_recursive_function(self, func_node: ast.FunctionDef) -> bool:
        """Check if a function is recursive."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_node.name:
                    return True
        return False
    
    def _has_memoization(self, func_node: ast.FunctionDef, content: str) -> bool:
        """Check if function has memoization (decorator or manual)."""
        # Check for decorators
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name) and 'cache' in decorator.id:
                return True
            if isinstance(decorator, ast.Attribute) and 'cache' in decorator.attr:
                return True
        # Also check for manual cache dictionary
        return 'cache' in content[func_node.lineno:func_node.body[0].lineno]
    
    def _calculate_complexity(self, func_node: ast.FunctionDef) -> int:
        """
        Calculate cyclomatic complexity of a function.
        A simplified version for quick analysis.
        """
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.With, ast.AsyncFor, ast.AsyncWith, ast.IfExp)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        return complexity

    def _estimate_function_lines(self, node: ast.FunctionDef, content: str) -> int:
        """Estimates the number of lines in a function."""
        if hasattr(node, 'end_lineno'):
            return node.end_lineno - node.lineno
        # Fallback for older python
        lines = content.splitlines()
        return len(lines[node.lineno-1 : node.body[-1].lineno])

    def _analyze_javascript_performance(self, file_path: str, content: str):
        """JavaScript-specific performance analysis."""
        # Similar to Python but with JS-specific patterns
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            # Check for common JS performance issues
            
            # Array operations in loops
            if 'for' in line and any(op in content[i:i+200] for op in ['.push(', '.unshift(']):
                self.issues.append(PerformanceIssue(
                    severity='low',
                    type='optimization',
                    location=f"{file_path}:{i+1}",
                    description="Array operations in loop",
                    impact="Repeated array operations can be slow",
                    suggestion="Consider using Array.map(), filter(), or building array once"
                ))
            
            # Document queries in loops
            if 'for' in line and any(query in content[i:i+200] for query in [
                'document.querySelector', 'document.getElementById', '$('
            ]):
                self.issues.append(PerformanceIssue(
                    severity='high',
                    type='dom',
                    location=f"{file_path}:{i+1}",
                    description="DOM queries inside loop",
                    impact="DOM operations are expensive - multiplied by loop iterations",
                    suggestion="Query once before loop and cache the result"
                ))
    
        # Regex-based checks for common JS/TS performance issues
        
        # 1. Nested loops
        # This is a very basic regex and prone to errors, but serves as a placeholder
        nested_loop_pattern = re.compile(r'for\s*\(.*\)\s*{[\s\S]*?for\s*\(.*\)')
        if nested_loop_pattern.search(content):
            self.issues.append(PerformanceIssue(
                severity='medium',
                type='algorithm',
                location=f"{file_path}",
                description="Nested loops detected in JavaScript/TypeScript",
                impact="Potential for O(n^2) complexity. Can slow down UI or server.",
                suggestion="Optimize loops. Use maps for lookups instead of nested iteration."
            ))
            self.analysis_stats['issues_found']['js_nested_loops'] += 1

        # 2. `useEffect` without dependency array in React
        use_effect_pattern = re.compile(r'useEffect\(\s*\(\)\s*=>\s*{[^}]*}(?!\s*,\s*\[))')
        if '.jsx' in file_path or '.tsx' in file_path:
            if use_effect_pattern.search(content):
                 self.issues.append(PerformanceIssue(
                    severity='high',
                    type='performance',
                    location=f"{file_path}",
                    description="`useEffect` hook with no dependency array",
                    impact="The effect will run after every single render, causing performance issues.",
                    suggestion="Add a dependency array `[]`. If the effect needs to run on updates, specify the dependencies."
                ))
                 self.analysis_stats['issues_found']['use_effect_no_deps'] += 1

    def _analyze_general_patterns(self, file_path: str, content: str, file_info: Dict[str, Any]):
        """Language-agnostic performance pattern detection."""
        lines = content.splitlines()
        
        # 1. TODO/FIXME comments
        todo_matches = re.findall(r'(TODO|FIXME):(.*)', content, re.IGNORECASE)
        for match in todo_matches:
            self.issues.append(PerformanceIssue(
                severity='low',
                type='maintenance',
                location=f"{file_path}",
                description=f"Found '{match[0]}' comment: {match[1].strip()}",
                impact="Indicates incomplete work or known issues that might affect stability.",
                suggestion="Address the TODO/FIXME or create a ticket to track it."
            ))
            self.analysis_stats['issues_found']['todo_comments'] += 1

        # 2. Large files
        if file_info.get('lines', 0) > 1000:
            self.issues.append(PerformanceIssue(
                severity='medium',
                type='maintainability',
                location=f"{file_path}",
                description=f"Very large file ({file_info.get('lines', 0)} lines)",
                impact="Large files are hard to understand, navigate, and maintain.",
                suggestion="Refactor into smaller, more focused modules."
            ))
            self.analysis_stats['issues_found']['large_files'] += 1

    def _analyze_architectural_performance(self, detailed_files: Dict[str, Any]):
        """
        Analyze cross-file patterns that might indicate performance issues.
        This looks at the bigger picture beyond individual files.
        """
        # 1. Circular Dependencies (basic check)
        try:
            dependency_graph = self._build_dependency_graph(detailed_files)
            circular_deps = self._find_circular_dependencies(dependency_graph)
            
            for cycle in circular_deps:
                self.issues.append(PerformanceIssue(
                    severity='high',
                    type='architecture',
                    location="Project-wide",
                    description=f"Circular dependency detected: {' -> '.join(cycle)}",
                    impact="Can lead to bugs, difficult testing, and problems with module initialization.",
                    suggestion="Break the cycle by using dependency inversion, interfaces, or event-based communication."
                ))
                self.analysis_stats['issues_found']['circular_dependency'] += 1
        except Exception:
            # Silently fail if graph construction has issues
            pass

    def _build_dependency_graph(self, detailed_files: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Build a graph of file dependencies based on imports.
        This is a simplified version and might need more robust handling for complex imports.
        """
        graph = defaultdict(list)
        file_map = {f: f for f in detailed_files.keys()}

        for file_path, info in detailed_files.items():
            if not info.get('imports'):
                continue
            
            current_file_base = Path(file_path).stem
            
            for imp in info['imports']:
                # This is a simplification. A real implementation would need to resolve relative paths.
                for other_file in file_map:
                    other_file_base = Path(other_file).stem
                    if imp == other_file_base or f".{imp}" in other_file_base:
                         if file_path != other_file:
                            graph[file_path].append(other_file)
        
        return graph

    def _find_circular_dependencies(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find circular dependencies in the dependency graph."""
        visiting = set()
        visited = set()
        cycles = []

        def dfs(node, path):
            visiting.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor in visiting:
                    # Cycle detected
                    cycle_start_index = path.index(neighbor)
                    cycles.append(path[cycle_start_index:] + [neighbor])
                elif neighbor not in visited:
                    dfs(neighbor, path)
            
            path.pop()
            visiting.remove(node)
            visited.add(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles

    def _calculate_performance_score(self):
        """Calculate overall performance score based on issues found."""
        score = 100
        for issue in self.issues:
            if issue.severity == 'critical':
                score -= 20
            elif issue.severity == 'high':
                score -= 10
            elif issue.severity == 'medium':
                score -= 5
            elif issue.severity == 'low':
                score -= 1
        
        self.performance_score = max(0, score)
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        # Group issues by type and severity
        issues_by_type = defaultdict(list)
        issues_by_severity = defaultdict(list)
        
        for issue in self.issues:
            issues_by_type[issue.type].append(issue)
            issues_by_severity[issue.severity].append(issue)
        
        # Generate insights
        insights = self._generate_performance_insights()
        
        # Create optimization roadmap
        roadmap = self._create_optimization_roadmap()
        
        return {
            'score': self.performance_score,
            'grade': self._score_to_grade(self.performance_score),
            'summary': {
                'total_issues': len(self.issues),
                'critical': len([i for i in self.issues if i.severity == 'critical']),
                'high': len([i for i in self.issues if i.severity == 'high']),
                'medium': len([i for i in self.issues if i.severity == 'medium']),
                'low': len([i for i in self.issues if i.severity == 'low']),
            },
            'issues': self.issues,
            'insights': self._generate_performance_insights(),
            'optimization_roadmap': self._create_optimization_roadmap(),
            'analysis_stats': self.analysis_stats
        }
    
    def _score_to_grade(self, score: int) -> str:
        """Convert numeric score to letter grade."""
        if score >= 95: return "A+"
        if score >= 90: return "A"
        if score >= 85: return "B+"
        if score >= 80: return "B"
        if score >= 75: return "B-"
        if score >= 70: return "C"
        if score >= 60: return "D"
        return "F"

    def _generate_performance_insights(self) -> List[str]:
        """Generate high-level insights from the analysis."""
        insights = []
        
        # Insight about most common issue
        if self.analysis_stats['issues_found']:
            most_common = max(self.analysis_stats['issues_found'].items(), 
                            key=lambda x: x[1])
            insights.append(
                f"🎯 Most common issue: {most_common[0].replace('_', ' ').title()} "
                f"({most_common[1]} occurrences)"
            )
        
        # Insight about critical issues
        critical_count = len([i for i in self.issues if i.severity == 'critical'])
        if critical_count > 0:
            insights.append(
                f"🚨 {critical_count} critical performance issues found that need immediate attention"
            )
        
        # Insight about algorithmic complexity
        nested_loops = self.analysis_stats['issues_found'].get('nested_loops', 0)
        if nested_loops > 0:
            insights.append(
                f"🔄 {nested_loops} nested loop structures found - consider algorithmic improvements"
            )
        
        # Insight about database performance
        n_plus_one = self.analysis_stats['issues_found'].get('n_plus_one', 0)
        if n_plus_one > 0:
            insights.append(
                f"🗄️ {n_plus_one} potential N+1 query problems detected - major database optimization opportunity"
            )
        
        # Overall health insight
        if self.performance_score >= 90:
            insights.append("✅ Overall performance health is excellent!")
        elif self.performance_score >= 70:
            insights.append("⚠️ Performance is acceptable but has room for improvement")
        else:
            insights.append("🚨 Significant performance optimizations needed")
        
        return insights
    
    def _create_optimization_roadmap(self) -> List[Dict[str, Any]]:
        """Create a prioritized roadmap for performance optimization."""
        roadmap = []
        
        # Group issues by impact and effort
        quick_wins = []
        major_improvements = []
        architectural_changes = []
        
        for issue in self.issues:
            effort = self._estimate_fix_effort(issue)
            impact = self._estimate_performance_impact(issue)
            
            item = {
                'issue': issue.description,
                'location': issue.location,
                'effort': effort,
                'impact': impact,
                'suggestion': issue.suggestion
            }
            
            if effort == 'low' and impact in ['high', 'medium']:
                quick_wins.append(item)
            elif impact == 'high':
                major_improvements.append(item)
            elif issue.type == 'architecture':
                architectural_changes.append(item)
        
        # Build roadmap
        if quick_wins:
            roadmap.append({
                'phase': 'Phase 1: Quick Wins',
                'description': 'Low effort, high impact optimizations',
                'timeline': '1-2 days',
                'items': quick_wins[:5]
            })
        
        if major_improvements:
            roadmap.append({
                'phase': 'Phase 2: Major Optimizations',
                'description': 'High impact performance improvements',
                'timeline': '1-2 weeks',
                'items': major_improvements[:5]
            })
        
        if architectural_changes:
            roadmap.append({
                'phase': 'Phase 3: Architectural Improvements',
                'description': 'Long-term performance and maintainability',
                'timeline': '1-2 months',
                'items': architectural_changes[:3]
            })
        
        return roadmap
    
    def _estimate_fix_effort(self, issue: PerformanceIssue) -> str:
        """Estimate the effort required to fix an issue."""
        if issue.type in ['optimization', 'memory'] and issue.severity == 'low':
            return 'low'
        elif issue.type == 'architecture' or issue.severity == 'critical':
            return 'high'
        else:
            return 'medium'
    
    def _estimate_performance_impact(self, issue: PerformanceIssue) -> str:
        """Estimate the performance impact of fixing an issue."""
        if issue.severity in ['critical', 'high']:
            return 'high'
        elif issue.severity == 'medium':
            return 'medium'
        else:
            return 'low'

class ContextAwarePerformancePredictor(PerformancePredictor):
    """Performance predictor that adjusts standards based on project context"""
    
    def __init__(self):
        super().__init__()
        self.project_context = {}
        self.performance_standards = {}
    
    def analyze_performance_with_context(self, detailed_files: Dict[str, Any], 
                                       project_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance with project-appropriate standards"""
        
        # First, understand what type of project this is
        self.project_context = self._classify_project_type(detailed_files, project_metadata)
        
        # Set appropriate performance standards
        self.performance_standards = self._get_context_appropriate_standards()
        
        # Run analysis with adjusted standards
        return self._analyze_with_adjusted_standards(detailed_files)
    
    def _classify_project_type(self, detailed_files: Dict[str, Any], 
                             metadata: Dict[str, Any]) -> Dict[str, str]:
        """Classify the project to understand appropriate performance expectations"""
        
        project_indicators = {
            'educational': 0,
            'production': 0,
            'research': 0,
            'prototype': 0
        }
        
        repo_name = metadata.get('repository', '').lower()
        technologies = metadata.get('technologies', [])
        total_lines = sum(f.get('lines', 0) for f in detailed_files.values())
        
        # Educational project indicators
        if any(keyword in repo_name for keyword in ['nano', 'mini', 'simple', 'tutorial', 'learn', 'demo']):
            project_indicators['educational'] += 50
        
        if total_lines < 2000:  # Small codebase
            project_indicators['educational'] += 30
            project_indicators['prototype'] += 20
        
        # Check for educational patterns in code
        for file_path, file_info in detailed_files.items():
            content = file_info.get('content_preview', '').lower()
            
            # Educational indicators
            if any(word in content for word in ['# simple', '# minimal', '# tutorial', '# example']):
                project_indicators['educational'] += 20
            
            # Production indicators
            if any(word in content for word in ['logging', 'error_handler', 'production', 'deploy']):
                project_indicators['production'] += 15
            
            # Research indicators  
            if any(word in content for word in ['experiment', 'paper', 'research', 'arxiv']):
                project_indicators['research'] += 20
        
        # Determine primary classification
        primary_type = max(project_indicators, key=project_indicators.get)
        
        return {
            'primary_type': primary_type,
            'confidence': project_indicators[primary_type],
            'secondary_indicators': project_indicators
        }
    
    def _get_context_appropriate_standards(self) -> Dict[str, Any]:
        """Get performance standards appropriate for the project type"""
        
        project_type = self.project_context.get('primary_type', 'production')
        
        standards = {
            'educational': {
                'complexity_threshold': 20,  # More lenient for educational code
                'function_size_threshold': 100,  # Larger functions OK for clarity
                'comment_ratio_minimum': 10,  # Lower comment requirements
                'duplication_tolerance': 15,  # Some duplication OK for learning
                'performance_weight': 0.3,  # Lower weight on performance
                'readability_weight': 0.7,  # Higher weight on readability
                'score_adjustment': 20  # Bonus points for educational projects
            },
            'production': {
                'complexity_threshold': 10,
                'function_size_threshold': 50,
                'comment_ratio_minimum': 20,
                'duplication_tolerance': 5,
                'performance_weight': 0.8,
                'readability_weight': 0.5,
                'score_adjustment': 0
            },
            'research': {
                'complexity_threshold': 15,  # Research code can be complex
                'function_size_threshold': 80,
                'comment_ratio_minimum': 15,
                'duplication_tolerance': 10,
                'performance_weight': 0.4,
                'readability_weight': 0.6,
                'score_adjustment': 10
            },
            'prototype': {
                'complexity_threshold': 15,
                'function_size_threshold': 75,
                'comment_ratio_minimum': 8,
                'duplication_tolerance': 20,
                'performance_weight': 0.4,
                'readability_weight': 0.6,
                'score_adjustment': 15
            }
        }
        
        return standards.get(project_type, standards['production'])
    
    def _analyze_with_adjusted_standards(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance analysis with context-appropriate standards"""
        
        # Reset issues and score for new analysis
        self.issues = []
        self.performance_score = 100
        
        standards = self.performance_standards
        project_type = self.project_context.get('primary_type', 'production')
        
        # Analyze each file with adjusted thresholds
        for file_path, file_info in detailed_files.items():
            if file_info.get('extension') in ['.py', '.js', '.ts', '.java']:
                self._analyze_file_with_context(file_path, file_info, standards)
        
        # Generate context-aware report
        return self._generate_context_aware_report(project_type, standards)
    
    def _analyze_file_with_context(self, file_path: str, file_info: Dict[str, Any], 
                                 standards: Dict[str, Any]):
        """Analyze a single file with context-appropriate standards"""
        
        content = file_info.get('full_content', file_info.get('content_preview', ''))
        
        if file_info.get('extension') == '.py':
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check complexity with adjusted threshold
                        complexity = self._calculate_complexity(node)
                        
                        if complexity > standards['complexity_threshold']:
                            severity = 'medium' if complexity < standards['complexity_threshold'] + 5 else 'high'
                            
                            self.issues.append(PerformanceIssue(
                                severity=severity,
                                type='complexity',
                                location=f"{file_path}:{node.lineno}",
                                description=f"Function '{node.name}' has complexity {complexity}",
                                impact=f"Complexity above {standards['complexity_threshold']} threshold for {self.project_context.get('primary_type')} projects",
                                suggestion=self._get_context_appropriate_suggestion('complexity', self.project_context.get('primary_type'))
                            ))
                        
                        # Check function size with adjusted threshold
                        func_lines = self._estimate_function_lines(node, content)
                        
                        if func_lines > standards['function_size_threshold']:
                            self.issues.append(PerformanceIssue(
                                severity='low',
                                type='maintainability',
                                location=f"{file_path}:{node.lineno}",
                                description=f"Large function '{node.name}' ({func_lines} lines)",
                                impact=f"Function size above {standards['function_size_threshold']} line threshold",
                                suggestion=self._get_context_appropriate_suggestion('function_size', self.project_context.get('primary_type'))
                            ))
                            
            except (SyntaxError, ValueError):
                pass
    
    def _get_context_appropriate_suggestion(self, issue_type: str, project_type: str) -> str:
        """Get suggestions appropriate for the project type"""
        
        suggestions = {
            'educational': {
                'complexity': "For educational code, consider adding more comments to explain complex logic rather than just reducing complexity.",
                'function_size': "Large functions can be OK for educational purposes if they improve understanding. Consider adding section comments."
            },
            'production': {
                'complexity': "Break down complex functions into smaller, testable units for production reliability.",
                'function_size': "Split large functions to improve maintainability and testing in production systems."
            },
            'research': {
                'complexity': "Complex research algorithms are acceptable, but add detailed documentation explaining the methodology.",
                'function_size': "Consider extracting helper functions while keeping main algorithm logic together for clarity."
            },
            'prototype': {
                'complexity': "For prototype code, complexity is acceptable if it speeds up development. Plan to refactor later.",
                'function_size': "Large functions are common in prototypes. Focus on functionality first, then refactor if it moves to production."
            }
        }
        
        return suggestions.get(project_type, {}).get(issue_type, "Consider refactoring for better maintainability.")
    
    def _generate_context_aware_report(self, project_type: str, standards: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report that considers project context"""
        
        # Calculate score with context adjustments
        base_score = 100
        
        for issue in self.issues:
            weight = 1.0
            
            # Adjust weights based on project type
            if project_type == 'educational':
                if issue.type in ['complexity', 'maintainability']:
                    weight *= 0.5  # Less penalty for educational projects
            elif project_type == 'research':
                if issue.type == 'complexity':
                    weight *= 0.7  # Research can be more complex
        
            # Apply severity penalties with weights
            if issue.severity == 'critical':
                base_score -= 20 * weight
            elif issue.severity == 'high':
                base_score -= 10 * weight
            elif issue.severity == 'medium':
                base_score -= 5 * weight
            elif issue.severity == 'low':
                base_score -= 2 * weight
        
        # Apply project type bonus
        final_score = min(100, base_score + standards.get('score_adjustment', 0))
        
        return {
            'score': max(0, final_score),
            'grade': self._score_to_grade(final_score),
            'project_context': self.project_context,
            'standards_used': standards,
            'context_message': self._generate_context_message(project_type),
            'issues': [issue.__dict__ for issue in self.issues],
            'issues_by_severity': self._group_issues_by_severity(),
            'recommendations': self._generate_context_recommendations(project_type)
        }
    
    def _group_issues_by_severity(self) -> Dict[str, List[PerformanceIssue]]:
        """Groups issues by severity."""
        grouped = defaultdict(list)
        for issue in self.issues:
            grouped[issue.severity].append(issue.__dict__)
        return grouped

    def _generate_context_message(self, project_type: str) -> str:
        """Generate a message explaining the context-aware analysis"""
        
        messages = {
            'educational': "This analysis uses educational project standards. Performance optimizations are less critical than code clarity and learning value.",
            'production': "This analysis uses production-grade standards with emphasis on performance, reliability, and maintainability.",
            'research': "This analysis uses research project standards, allowing for higher complexity in favor of algorithmic clarity.",
            'prototype': "This analysis uses prototype standards, balancing rapid development with basic quality practices."
        }
        
        return messages.get(project_type, "Standard analysis applied.")
    
    def _generate_context_recommendations(self, project_type: str) -> list[str]:
        """ Generate recommendations based on project context """
        recs = {
            'educational': [
                "Focus on adding detailed comments to explain complex parts of the code for future learners.",
                "Ensure variable and function names are descriptive and easy to understand.",
                "It's okay to have larger functions if it helps demonstrate a concept from start to finish."
            ],
            'production': [
                "Prioritize fixing high-severity performance issues to ensure user satisfaction.",
                "Add comprehensive logging and error handling for all critical paths.",
                "Invest in automated testing (unit, integration) to maintain stability."
            ],
            'research': [
                "Document the algorithms and data structures used with references to relevant papers.",
                "Ensure the research code is reproducible by managing dependencies and data.",
                "Code clarity is important, but complexity is acceptable if it's essential for the research."
            ],
             'prototype': [
                "Focus on delivering core functionality quickly. Don't over-engineer.",
                "Use 'TODO' comments to mark areas that need technical debt cleanup before production.",
                "Avoid premature optimization. Build it first, then measure and optimize if needed."
            ]
        }
        return recs.get(project_type, [])