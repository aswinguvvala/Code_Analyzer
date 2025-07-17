# code_evolution_analyzer.py
import git
import ast
import difflib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import hashlib
import re

class CodeEvolutionAnalyzer:
    """
    This analyzer creates a 'time machine' for your codebase, allowing us to
    understand not just what the code is, but how it got there.
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)
        self.evolution_data = {
            'functions': defaultdict(list),  # Track function evolution
            'files': defaultdict(list),      # Track file evolution
            'complexity': defaultdict(list),  # Track complexity over time
            'architecture': [],              # Track architectural changes
            'bug_patterns': [],              # Detected bug patterns
            'refactoring_opportunities': []  # Predicted refactoring needs
        }
    
    def analyze_evolution(self, months_back: int = 6, sample_rate: int = 7) -> Dict[str, Any]:
        """
        Analyze repository evolution over time.
        
        Args:
            months_back: How many months of history to analyze
            sample_rate: Analyze every Nth commit to balance speed/detail
        
        Returns:
            Comprehensive evolution analysis with predictions
        """
        print(f"🕰️ Analyzing {months_back} months of evolution...")
        
        # Get commits from the time period
        since_date = datetime.now() - timedelta(days=months_back * 30)
        commits = list(self.repo.iter_commits('HEAD', since=since_date))
        
        # Sample commits to analyze (analyzing every commit would be slow)
        sampled_commits = commits[::sample_rate]
        
        print(f"📊 Analyzing {len(sampled_commits)} commits out of {len(commits)} total")
        
        # Analyze each sampled commit
        for i, commit in enumerate(sampled_commits):
            print(f"⏳ Analyzing commit {i+1}/{len(sampled_commits)}: {commit.hexsha[:8]}")
            self._analyze_commit(commit)
        
        # Perform higher-level analysis
        self._detect_growth_patterns()
        self._identify_bug_patterns()
        self._detect_architecture_drift()
        self._predict_refactoring_needs()
        
        return self._generate_evolution_report()
    
    def _analyze_commit(self, commit: git.Commit):
        """
        Analyze a single commit for evolution data.
        This is where we extract the 'DNA' of each code snapshot.
        """
        # Temporarily checkout this commit to analyze it
        original_head = self.repo.head.commit
        
        try:
            self.repo.head.reference = commit
            self.repo.head.reset(index=True, working_tree=True)
            
            # Analyze Python files at this point in time
            for item in self.repo.tree().traverse():
                if item.type == 'blob' and item.path.endswith('.py'):
                    self._analyze_file_at_commit(item.path, commit)
            
        finally:
            # Always return to original commit
            self.repo.head.reference = original_head
            self.repo.head.reset(index=True, working_tree=True)
    
    def _analyze_file_at_commit(self, file_path: str, commit: git.Commit):
        """
        Analyze a specific file at a specific point in time.
        We track how functions grow, complexity changes, and patterns emerge.
        """
        try:
            # Read file content at this commit
            file_content = self.repo.oid_to_object(
                commit.tree[file_path].binsha
            ).data_stream.read().decode('utf-8')
            
            # Parse the Python code
            tree = ast.parse(file_content)
            
            # Track file size evolution
            self.evolution_data['files'][file_path].append({
                'commit': commit.hexsha[:8],
                'date': commit.committed_datetime,
                'lines': len(file_content.splitlines()),
                'size_bytes': len(file_content.encode('utf-8'))
            })
            
            # Analyze each function in the file
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._track_function_evolution(node, file_path, commit, file_content)
                    
        except Exception as e:
            # File might not exist at this commit or might have syntax errors
            pass
    
    def _track_function_evolution(self, func_node: ast.FunctionDef, file_path: str, 
                                  commit: git.Commit, full_content: str):
        """
        Track how individual functions evolve over time.
        This helps us identify functions that are growing out of control.
        """
        # Create a unique identifier for the function
        func_id = f"{file_path}::{func_node.name}"
        
        # Calculate function metrics
        func_lines = self._count_function_lines(func_node, full_content)
        complexity = self._calculate_complexity(func_node)
        
        # Store evolution data
        self.evolution_data['functions'][func_id].append({
            'commit': commit.hexsha[:8],
            'date': commit.committed_datetime,
            'lines': func_lines,
            'complexity': complexity,
            'parameters': len(func_node.args.args),
            'has_docstring': ast.get_docstring(func_node) is not None
        })
        
        # Track complexity evolution for the entire file
        self.evolution_data['complexity'][file_path].append({
            'commit': commit.hexsha[:8],
            'date': commit.committed_datetime,
            'function': func_node.name,
            'complexity': complexity
        })
    
    def _count_function_lines(self, func_node: ast.FunctionDef, full_content: str) -> int:
        """Count lines in a function by analyzing the AST node positions."""
        lines = full_content.splitlines()
        if hasattr(func_node, 'end_lineno'):
            return func_node.end_lineno - func_node.lineno + 1
        else:
            # Fallback for older Python versions
            return len(lines[func_node.lineno - 1:])
    
    def _calculate_complexity(self, func_node: ast.FunctionDef) -> int:
        """
        Calculate cyclomatic complexity of a function.
        Each decision point adds to complexity.
        """
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            # Each of these adds a decision path
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # 'and' and 'or' operations add complexity
                complexity += len(node.values) - 1
                
        return complexity
    
    def _detect_growth_patterns(self):
        """
        Identify functions and files that are growing rapidly.
        These are prime candidates for refactoring.
        """
        growth_alerts = []
        
        # Analyze function growth
        for func_id, evolution in self.evolution_data['functions'].items():
            if len(evolution) < 2:
                continue
                
            # Calculate growth rate
            start_lines = evolution[0]['lines']
            end_lines = evolution[-1]['lines']
            
            if start_lines > 0:
                growth_rate = ((end_lines - start_lines) / start_lines) * 100
                
                if growth_rate > 200:  # Function tripled in size
                    growth_alerts.append({
                        'type': 'rapid_growth',
                        'function': func_id,
                        'growth_rate': growth_rate,
                        'start_lines': start_lines,
                        'end_lines': end_lines,
                        'recommendation': f"This function has grown {growth_rate:.0f}%. Consider breaking it into smaller functions."
                    })
                    
            # Check complexity growth
            start_complexity = evolution[0]['complexity']
            end_complexity = evolution[-1]['complexity']
            
            if end_complexity > 10 and end_complexity > start_complexity * 1.5:
                growth_alerts.append({
                    'type': 'complexity_growth',
                    'function': func_id,
                    'start_complexity': start_complexity,
                    'end_complexity': end_complexity,
                    'recommendation': "Complexity has increased significantly. Simplify control flow."
                })
        
        self.evolution_data['growth_patterns'] = growth_alerts
    
    def _identify_bug_patterns(self):
        """
        Detect patterns in bug fixes by analyzing commit messages and changes.
        This helps identify error-prone areas of the code.
        """
        bug_patterns = []
        
        # Look for bug-fixing commits
        bug_keywords = ['fix', 'bug', 'error', 'issue', 'problem', 'crash', 'fault']
        
        for commit in self.repo.iter_commits('HEAD', max_count=500):
            commit_msg_lower = commit.message.lower()
            
            if any(keyword in commit_msg_lower for keyword in bug_keywords):
                # Analyze what files were changed in this bug fix
                changed_files = list(commit.stats.files.keys())
                
                for file_path in changed_files:
                    if file_path.endswith('.py'):
                        # Track this as a bug-prone file
                        bug_patterns.append({
                            'commit': commit.hexsha[:8],
                            'date': commit.committed_datetime,
                            'file': file_path,
                            'message': commit.message.split('\n')[0],  # First line only
                            'changes': commit.stats.files[file_path]
                        })
        
        # Identify frequently fixed files
        file_bug_counts = defaultdict(int)
        for pattern in bug_patterns:
            file_bug_counts[pattern['file']] += 1
        
        # Find files with recurring bugs
        problematic_files = []
        for file_path, bug_count in file_bug_counts.items():
            if bug_count >= 3:  # Three or more bug fixes
                problematic_files.append({
                    'file': file_path,
                    'bug_fixes': bug_count,
                    'recommendation': f"This file has been fixed {bug_count} times. Consider a thorough review or rewrite."
                })
        
        self.evolution_data['bug_patterns'] = problematic_files
    
    def _detect_architecture_drift(self):
        """
        Detect how the codebase architecture has changed over time.
        This helps identify when the original design is being violated.
        """
        architecture_changes = []
        
        # Sample commits to check architecture
        commits = list(self.repo.iter_commits('HEAD', max_count=100))
        sample_points = [commits[i] for i in [0, len(commits)//2, -1] if i < len(commits)]
        
        for commit in sample_points:
            # Analyze directory structure at this point
            structure = self._analyze_directory_structure(commit)
            architecture_changes.append({
                'commit': commit.hexsha[:8],
                'date': commit.committed_datetime,
                'structure': structure
            })
        
        # Detect drift by comparing structures
        if len(architecture_changes) >= 2:
            original = architecture_changes[-1]['structure']
            current = architecture_changes[0]['structure']
            
            # Check for new top-level modules (possible architecture violation)
            new_modules = set(current.keys()) - set(original.keys())
            if new_modules:
                self.evolution_data['architecture'].append({
                    'type': 'new_modules',
                    'modules': list(new_modules),
                    'recommendation': "New top-level modules detected. Ensure they align with architectural principles."
                })
    
    def _analyze_directory_structure(self, commit: git.Commit) -> Dict[str, int]:
        """Analyze the directory structure at a specific commit."""
        structure = defaultdict(int)
        
        try:
            for item in commit.tree.traverse():
                if item.type == 'blob' and item.path.endswith('.py'):
                    # Count files per directory
                    directory = item.path.split('/')[0] if '/' in item.path else 'root'
                    structure[directory] += 1
        except:
            pass
            
        return dict(structure)
    
    def _predict_refactoring_needs(self):
        """
        Use evolution data to predict where refactoring will be needed.
        This is the 'crystal ball' feature - predicting future problems.
        """
        predictions = []
        
        # Analyze function growth trends
        for func_id, evolution in self.evolution_data['functions'].items():
            if len(evolution) < 3:  # Need at least 3 data points for trend
                continue
            
            # Calculate growth trajectory
            recent_growth = self._calculate_growth_trend(evolution[-3:])
            
            if recent_growth > 20:  # 20% growth per sample
                # Predict future size
                current_lines = evolution[-1]['lines']
                predicted_lines = current_lines * (1 + recent_growth/100) ** 3  # 3 samples ahead
                
                if predicted_lines > 100:  # Will become too large
                    predictions.append({
                        'function': func_id,
                        'current_lines': current_lines,
                        'predicted_lines': int(predicted_lines),
                        'recommendation': f"At current growth rate, this function will reach {int(predicted_lines)} lines soon. Refactor now to prevent future problems.",
                        'priority': 'high' if predicted_lines > 150 else 'medium'
                    })
        
        self.evolution_data['refactoring_opportunities'] = predictions
    
    def _calculate_growth_trend(self, evolution_points: List[Dict]) -> float:
        """Calculate average growth rate from evolution data points."""
        if len(evolution_points) < 2:
            return 0
        
        total_growth = 0
        for i in range(1, len(evolution_points)):
            prev_lines = evolution_points[i-1]['lines']
            curr_lines = evolution_points[i]['lines']
            
            if prev_lines > 0:
                growth = ((curr_lines - prev_lines) / prev_lines) * 100
                total_growth += growth
        
        return total_growth / (len(evolution_points) - 1)
    
    def _generate_evolution_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evolution report with visualizations and insights.
        This is what gets displayed to the user.
        """
        report = {
            'summary': self._generate_summary(),
            'growth_alerts': self.evolution_data.get('growth_patterns', []),
            'bug_patterns': self.evolution_data.get('bug_patterns', []),
            'architecture_changes': self.evolution_data.get('architecture', []),
            'refactoring_predictions': self.evolution_data.get('refactoring_opportunities', []),
            'timeline_data': self._prepare_timeline_data(),
            'insights': self._generate_insights()
        }
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate high-level summary statistics."""
        total_functions = len(self.evolution_data['functions'])
        growing_functions = len([f for f in self.evolution_data.get('growth_patterns', []) 
                               if f['type'] == 'rapid_growth'])
        
        return {
            'functions_tracked': total_functions,
            'rapidly_growing_functions': growing_functions,
            'bug_prone_files': len(self.evolution_data.get('bug_patterns', [])),
            'refactoring_needed': len(self.evolution_data.get('refactoring_opportunities', []))
        }
    
    def _prepare_timeline_data(self) -> Dict[str, Any]:
        """
        Prepare data for timeline visualizations.
        This creates the data structure needed for charts.
        """
        timeline_data = {
            'file_growth': [],
            'complexity_evolution': [],
            'bug_fix_timeline': []
        }
        
        # Aggregate file growth data
        for file_path, evolution in self.evolution_data['files'].items():
            if evolution:
                timeline_data['file_growth'].append({
                    'file': file_path,
                    'data_points': [(e['date'].isoformat(), e['lines']) for e in evolution]
                })
        
        # Aggregate complexity data
        for file_path, complexity_data in self.evolution_data['complexity'].items():
            if complexity_data:
                # Average complexity per commit
                avg_by_commit = defaultdict(list)
                for entry in complexity_data:
                    avg_by_commit[entry['commit']].append(entry['complexity'])
                
                timeline_data['complexity_evolution'].append({
                    'file': file_path,
                    'data_points': [(entry['date'].isoformat(), 
                                   sum(avg_by_commit[entry['commit']]) / len(avg_by_commit[entry['commit']]))
                                   for entry in complexity_data if entry['commit'] in avg_by_commit]
                })
        
        return timeline_data
    
    def _generate_insights(self) -> List[str]:
        """
        Generate human-readable insights from the evolution analysis.
        These are the 'aha!' moments we want to surface to developers.
        """
        insights = []
        
        # Insight about rapid growth
        growth_patterns = self.evolution_data.get('growth_patterns', [])
        if growth_patterns:
            worst_growth = max(growth_patterns, key=lambda x: x.get('growth_rate', 0))
            insights.append(
                f"🚨 The function '{worst_growth['function']}' has grown {worst_growth['growth_rate']:.0f}% "
                f"- this is a critical refactoring candidate!"
            )
        
        # Insight about bug patterns
        bug_patterns = self.evolution_data.get('bug_patterns', [])
        if bug_patterns:
            worst_file = max(bug_patterns, key=lambda x: x['bug_fixes'])
            insights.append(
                f"🐛 The file '{worst_file['file']}' has required {worst_file['bug_fixes']} bug fixes. "
                f"Consider a comprehensive review to address underlying issues."
            )
        
        # Insight about predictions
        predictions = self.evolution_data.get('refactoring_opportunities', [])
        high_priority = [p for p in predictions if p.get('priority') == 'high']
        if high_priority:
            insights.append(
                f"🔮 Based on growth trends, {len(high_priority)} functions will become unmaintainable "
                f"within the next few months. Proactive refactoring recommended!"
            )
        
        # General health insight
        if not growth_patterns and not bug_patterns:
            insights.append(
                "✅ Your codebase evolution looks healthy! No major growth or bug patterns detected."
            )
        
        return insights