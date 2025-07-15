# code_quality_analyzer.py - New module to add to your project

import ast
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path
import hashlib

class CodeQualityAnalyzer:
    """
    This class analyzes code quality metrics to help developers understand
    the health and maintainability of their codebase.
    
    Think of this as a fitness tracker for your code - it measures various
    health indicators to help you spot potential problems.
    """
    
    def __init__(self):
        # These thresholds are based on software engineering best practices
        self.complexity_thresholds = {
            'low': 5,      # Simple functions, easy to test
            'medium': 10,  # Moderate complexity, still manageable  
            'high': 15,    # Complex, should consider refactoring
            'very_high': 20 # Very complex, definitely needs attention
        }
    
    def calculate_cyclomatic_complexity(self, content: str, file_ext: str) -> Dict[str, Any]:
        """
        Cyclomatic complexity measures how many different paths your code can take.
        Think of it like counting how many different routes you could take through a city.
        More routes = more complexity = harder to test and maintain.
        
        For example:
        - A simple function with no if/while statements = complexity 1
        - Add one if statement = complexity 2 (two possible paths)
        - Add a while loop = complexity 3, and so on
        """
        if file_ext != '.py':
            return {'average_complexity': 0, 'functions': [], 'total_complexity': 0}
        
        try:
            # Parse the Python code into an Abstract Syntax Tree
            # This lets us analyze the structure without executing the code
            tree = ast.parse(content)
            function_complexities = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count decision points in each function
                    complexity = self._calculate_function_complexity(node)
                    function_complexities.append({
                        'name': node.name,
                        'complexity': complexity,
                        'line': node.lineno,
                        'risk_level': self._get_complexity_risk_level(complexity)
                    })
            
            # Calculate statistics
            total_complexity = sum(f['complexity'] for f in function_complexities)
            avg_complexity = total_complexity / len(function_complexities) if function_complexities else 0
            
            return {
                'average_complexity': round(avg_complexity, 2),
                'total_complexity': total_complexity,
                'functions': function_complexities,
                'high_complexity_functions': [f for f in function_complexities if f['complexity'] > self.complexity_thresholds['medium']]
            }
            
        except Exception as e:
            return {'error': str(e), 'average_complexity': 0, 'functions': []}
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """
        Count the number of decision points in a function.
        Each if, while, for, try, except, and, or increases complexity by 1.
        """
        complexity = 1  # Base complexity for any function
        
        for node in ast.walk(func_node):
            # These are all decision points that create different execution paths
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.BoolOp, ast.Compare)):
                # Boolean operations (and, or) add branching complexity
                complexity += 1
        
        return complexity
    
    def _get_complexity_risk_level(self, complexity: int) -> str:
        """Categorize complexity level for easy understanding"""
        if complexity <= self.complexity_thresholds['low']:
            return 'low'
        elif complexity <= self.complexity_thresholds['medium']:
            return 'medium'
        elif complexity <= self.complexity_thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def detect_code_duplication(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find similar code blocks across files.
        Like finding copy-paste code that should probably be extracted into functions.
        
        We use a technique called "hashing" - converting code blocks into
        fingerprints that we can compare quickly.
        """
        code_hashes = {}  # Store fingerprints of code blocks
        duplicates = []
        
        for file_path, file_info in detailed_files.items():
            if not file_path.endswith(('.py', '.js', '.ts', '.java')):
                continue
                
            content = file_info.get('content_preview', '')
            if not content:
                continue
            
            # Split code into logical blocks (functions, classes, etc.)
            blocks = self._extract_code_blocks(content, file_info.get('extension', ''))
            
            for block_info in blocks:
                # Create a "fingerprint" of the code block
                normalized_block = self._normalize_code_block(block_info['content'])
                block_hash = hashlib.md5(normalized_block.encode()).hexdigest()
                
                if block_hash in code_hashes:
                    # Found a duplicate!
                    duplicates.append({
                        'original_file': code_hashes[block_hash]['file'],
                        'original_location': code_hashes[block_hash]['location'],
                        'duplicate_file': file_path,
                        'duplicate_location': block_info['location'],
                        'similarity_score': 100,  # Exact match
                        'lines': block_info['lines']
                    })
                else:
                    code_hashes[block_hash] = {
                        'file': file_path,
                        'location': block_info['location'],
                        'content': block_info['content']
                    }
        
        return {
            'total_duplicates': len(duplicates),
            'duplicate_pairs': duplicates,
            'duplication_percentage': self._calculate_duplication_percentage(duplicates, detailed_files)
        }
    
    def _extract_code_blocks(self, content: str, file_ext: str) -> List[Dict[str, Any]]:
        """
        Break code into logical chunks that we can compare.
        For Python, we extract functions and classes.
        For other languages, we use simpler line-based blocks.
        """
        blocks = []
        lines = content.split('\n')
        
        if file_ext == '.py':
            # For Python, try to extract functions and classes
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        start_line = node.lineno
                        # Estimate end line (simplified approach)
                        end_line = min(start_line + 20, len(lines))
                        block_content = '\n'.join(lines[start_line-1:end_line])
                        
                        blocks.append({
                            'content': block_content,
                            'location': f"lines {start_line}-{end_line}",
                            'lines': end_line - start_line + 1,
                            'type': 'function' if isinstance(node, ast.FunctionDef) else 'class'
                        })
            except:
                # Fall back to simple line blocks if parsing fails
                pass
        
        # For non-Python files or as fallback, use simple line-based blocks
        if not blocks:
            for i in range(0, len(lines), 10):  # 10-line blocks
                block_lines = lines[i:i+10]
                if len([l for l in block_lines if l.strip()]) > 3:  # Skip mostly empty blocks
                    blocks.append({
                        'content': '\n'.join(block_lines),
                        'location': f"lines {i+1}-{i+len(block_lines)}",
                        'lines': len(block_lines),
                        'type': 'block'
                    })
        
        return blocks
    
    def _normalize_code_block(self, content: str) -> str:
        """
        Normalize code for comparison by removing comments and extra whitespace.
        This helps us find code that's functionally the same even if formatting differs.
        """
        # Remove comments
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)  # Python comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)  # JS/Java comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # Block comments
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def _calculate_duplication_percentage(self, duplicates: List[Dict], detailed_files: Dict) -> float:
        """Calculate what percentage of the codebase is duplicated"""
        if not duplicates or not detailed_files:
            return 0.0
        
        total_lines = sum(f.get('lines', 0) for f in detailed_files.values())
        duplicate_lines = sum(d['lines'] for d in duplicates)
        
        return round((duplicate_lines / max(total_lines, 1)) * 100, 2)
    
    def analyze_function_sizes(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze function sizes to identify overly large functions.
        Large functions are harder to understand, test, and maintain.
        
        Best practice: Functions should typically be 20-50 lines max.
        """
        function_stats = {
            'total_functions': 0,
            'large_functions': [],  # Functions over 50 lines
            'average_size': 0,
            'size_distribution': {'small': 0, 'medium': 0, 'large': 0, 'very_large': 0}
        }
        
        all_function_sizes = []
        
        for file_path, file_info in detailed_files.items():
            if not file_path.endswith('.py'):
                continue
                
            content = file_info.get('content_preview', '')
            if not content:
                continue
            
            try:
                tree = ast.parse(content)
                lines = content.split('\n')
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Estimate function size
                        start_line = node.lineno
                        end_line = start_line
                        
                        # Find the end of the function by looking for the next function or class
                        for other_node in ast.walk(tree):
                            if (isinstance(other_node, (ast.FunctionDef, ast.ClassDef)) and 
                                other_node.lineno > start_line and 
                                (end_line == start_line or other_node.lineno < end_line)):
                                end_line = other_node.lineno - 1
                        
                        if end_line == start_line:
                            end_line = len(lines)
                        
                        function_size = end_line - start_line + 1
                        all_function_sizes.append(function_size)
                        function_stats['total_functions'] += 1
                        
                        # Categorize function size
                        if function_size <= 20:
                            function_stats['size_distribution']['small'] += 1
                        elif function_size <= 50:
                            function_stats['size_distribution']['medium'] += 1
                        elif function_size <= 100:
                            function_stats['size_distribution']['large'] += 1
                            function_stats['large_functions'].append({
                                'name': node.name,
                                'file': file_path,
                                'lines': function_size,
                                'start_line': start_line
                            })
                        else:
                            function_stats['size_distribution']['very_large'] += 1
                            function_stats['large_functions'].append({
                                'name': node.name,
                                'file': file_path,
                                'lines': function_size,
                                'start_line': start_line
                            })
                            
            except Exception:
                continue
        
        if all_function_sizes:
            function_stats['average_size'] = round(sum(all_function_sizes) / len(all_function_sizes), 2)
        
        return function_stats
    
    def calculate_comment_ratio(self, content: str, file_ext: str) -> float:
        """
        Calculate the ratio of comment lines to code lines.
        Good code typically has 10-30% comments.
        """
        lines = content.split('\n')
        comment_lines = 0
        code_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Check if line is a comment
            if file_ext == '.py' and stripped.startswith('#'):
                comment_lines += 1
            elif file_ext in ['.js', '.ts', '.java'] and (stripped.startswith('//') or stripped.startswith('/*')):
                comment_lines += 1
            else:
                code_lines += 1
        
        if code_lines == 0:
            return 0.0
        
        return round((comment_lines / (comment_lines + code_lines)) * 100, 2)
    
    def generate_quality_score(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an overall quality score based on various metrics.
        Think of this like a credit score for your code!
        """
        score = 100  # Start with perfect score
        factors = []
        
        # Deduct points for high complexity
        avg_complexity = quality_metrics.get('complexity', {}).get('average_complexity', 0)
        if avg_complexity > 10:
            deduction = min(30, (avg_complexity - 10) * 3)
            score -= deduction
            factors.append(f"High complexity (-{deduction} points)")
        
        # Deduct points for code duplication
        duplication_pct = quality_metrics.get('duplication', {}).get('duplication_percentage', 0)
        if duplication_pct > 5:
            deduction = min(25, duplication_pct * 2)
            score -= deduction
            factors.append(f"Code duplication (-{deduction} points)")
        
        # Deduct points for large functions
        large_funcs = len(quality_metrics.get('function_sizes', {}).get('large_functions', []))
        if large_funcs > 0:
            deduction = min(20, large_funcs * 5)
            score -= deduction
            factors.append(f"Large functions (-{deduction} points)")
        
        # Deduct points for poor commenting
        comment_ratio = quality_metrics.get('comment_ratio', 15)
        if comment_ratio < 5:
            score -= 10
            factors.append("Low comment ratio (-10 points)")
        elif comment_ratio > 50:
            score -= 5
            factors.append("Excessive comments (-5 points)")
        
        score = max(0, score)  # Don't go below 0
        
        # Determine grade
        if score >= 90:
            grade = 'A'
        elif score >= 80:
            grade = 'B'
        elif score >= 70:
            grade = 'C'
        elif score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'score': score,
            'grade': grade,
            'factors': factors,
            'recommendations': self._generate_recommendations(quality_metrics)
        }
    
    def _generate_recommendations(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations for improving code quality"""
        recommendations = []
        
        complexity = quality_metrics.get('complexity', {})
        if complexity.get('average_complexity', 0) > 10:
            recommendations.append("🔄 Consider breaking down complex functions into smaller, more focused functions")
        
        duplication = quality_metrics.get('duplication', {})
        if duplication.get('total_duplicates', 0) > 0:
            recommendations.append("♻️ Extract duplicated code into reusable functions or modules")
        
        function_sizes = quality_metrics.get('function_sizes', {})
        large_funcs = function_sizes.get('large_functions', [])
        if large_funcs:
            recommendations.append(f"📏 Consider splitting {len(large_funcs)} large functions into smaller ones")
        
        comment_ratio = quality_metrics.get('comment_ratio', 15)
        if comment_ratio < 10:
            recommendations.append("📝 Add more comments to explain complex logic and business rules")
        
        if not recommendations:
            recommendations.append("✨ Great job! Your code quality metrics look good")
        
        return recommendations