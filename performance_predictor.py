# performance_predictor.py
import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

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
                    
        except SyntaxError:
            # Skip files with syntax errors
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
                                suggestion="Use list.append() and ''.join() instead, or io.StringIO for large strings"
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
                
                self.issues.append(PerformanceIssue(
                    severity='low',
                    type='optimization',
                    location=f"{file_path}:{node.lineno}",
                    description="Loop could be replaced with list comprehension",
                    impact="List comprehensions are ~30% faster than equivalent loops",
                    suggestion="Replace with: [expression for item in iterable]"
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
    
    def _analyze_function_performance(self, func_node: ast.FunctionDef, file_path: str, content: str):
        """
        Analyze individual function performance characteristics.
        """
        # Check for recursive functions without memoization
        if self._is_recursive_function(func_node):
            # Check if it has memoization
            has_memoization = self._has_memoization(func_node, content)
            
            if not has_memoization:
                self.issues.append(PerformanceIssue(
                    severity='medium',
                    type='algorithm',
                    location=f"{file_path}:{func_node.lineno}",
                    description=f"Recursive function '{func_node.name}' without memoization",
                    impact="Exponential time complexity for overlapping subproblems",
                    suggestion="Add @functools.lru_cache() decorator or implement memoization"
                ))
                self.analysis_stats['issues_found']['unmemoized_recursion'] += 1
        
        # Check function complexity
        complexity = self._calculate_function_complexity(func_node)
        if complexity > 20:
            self.issues.append(PerformanceIssue(
                severity='low',
                type='maintainability',
                location=f"{file_path}:{func_node.lineno}",
                description=f"High complexity function (complexity: {complexity})",
                impact="Complex functions are harder to optimize and maintain",
                suggestion="Consider breaking into smaller functions"
            ))
    
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
            dec_str = ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator)
            if 'cache' in dec_str or 'memoize' in dec_str:
                return True
        
        # Check for manual memoization pattern
        func_body = ast.unparse(func_node) if hasattr(ast, 'unparse') else ""
        return 'cache' in func_body and ('{}' in func_body or 'dict()' in func_body)
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
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
    
    def _analyze_general_patterns(self, file_path: str, content: str, file_info: Dict[str, Any]):
        """Language-agnostic performance pattern detection."""
        lines = content.splitlines()
        
        # Large function detection
        functions = file_info.get('functions', [])
        for func in functions:
            # Estimate function size (rough)
            func_lines = len([l for l in lines if func in l])
            if func_lines > 100:
                self.issues.append(PerformanceIssue(
                    severity='low',
                    type='maintainability',
                    location=f"{file_path}",
                    description=f"Large function: {func} (~{func_lines} lines)",
                    impact="Large functions are harder to optimize and test",
                    suggestion="Break into smaller, focused functions"
                ))
        
        # Regex compilation in loops/functions
        regex_patterns = re.findall(r're\.(compile|search|match|findall)\s*\(', content)
        if len(regex_patterns) > 5:
            self.issues.append(PerformanceIssue(
                severity='medium',
                type='optimization',
                location=file_path,
                description="Multiple regex operations detected",
                impact="Regex compilation is expensive if done repeatedly",
                suggestion="Pre-compile regex patterns at module level"
            ))
    
    def _analyze_architectural_performance(self, detailed_files: Dict[str, Any]):
        """
        Analyze cross-file patterns that might indicate performance issues.
        This looks at the bigger picture beyond individual files.
        """
        # Check for circular dependencies
        dependency_graph = self._build_dependency_graph(detailed_files)
        cycles = self._find_circular_dependencies(dependency_graph)
        
        if cycles:
            self.issues.append(PerformanceIssue(
                severity='medium',
                type='architecture',
                location='Multiple files',
                description=f"Circular dependencies detected between {len(cycles)} modules",
                impact="Can cause import performance issues and make code harder to optimize",
                suggestion="Refactor to break circular dependencies - consider dependency injection"
            ))
        
        # Check for monolithic files
        large_files = []
        for file_path, file_info in detailed_files.items():
            if file_info.get('lines', 0) > 1000:
                large_files.append((file_path, file_info['lines']))
        
        if large_files:
            worst_file = max(large_files, key=lambda x: x[1])
            self.issues.append(PerformanceIssue(
                severity='low',
                type='architecture',
                location=worst_file[0],
                description=f"Very large file with {worst_file[1]} lines",
                impact="Large files load slower and are harder to optimize",
                suggestion="Consider splitting into multiple modules"
            ))
    
    def _build_dependency_graph(self, detailed_files: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build a graph of file dependencies based on imports."""
        graph = defaultdict(list)
        
        for file_path, file_info in detailed_files.items():
            imports = file_info.get('imports', [])
            for imp in imports:
                # Extract module name from import statement
                # This is simplified - real implementation would be more robust
                if 'from' in imp:
                    module = imp.split('from')[1].split('import')[0].strip()
                else:
                    module = imp.replace('import', '').strip().split('.')[0]
                
                # Check if it's a local import
                for other_file in detailed_files:
                    if module in other_file:
                        graph[file_path].append(other_file)
                        break
        
        return dict(graph)
    
    def _find_circular_dependencies(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find circular dependencies in the dependency graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _calculate_performance_score(self):
        """Calculate overall performance score based on issues found."""
        for issue in self.issues:
            if issue.severity == 'critical':
                self.performance_score -= 20
            elif issue.severity == 'high':
                self.performance_score -= 10
            elif issue.severity == 'medium':
                self.performance_score -= 5
            elif issue.severity == 'low':
                self.performance_score -= 2
        
        self.performance_score = max(0, self.performance_score)
    
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
                'critical_issues': len(issues_by_severity['critical']),
                'high_issues': len(issues_by_severity['high']),
                'files_analyzed': self.analysis_stats['files_analyzed'],
                'functions_analyzed': self.analysis_stats['functions_analyzed']
            },
            'issues': self.issues,
            'issues_by_type': dict(issues_by_type),
            'issues_by_severity': dict(issues_by_severity),
            'insights': insights,
            'optimization_roadmap': roadmap,
            'detailed_stats': dict(self.analysis_stats['issues_found'])
        }
    
    def _score_to_grade(self, score: int) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
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