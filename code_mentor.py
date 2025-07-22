import asyncio
import streamlit as st
from typing import Dict, List, Any, Optional
from pathlib import Path

class InteractiveCodeMentor:
    """
    Interactive code mentor for helping users understand code analysis results.
    This class provides intelligent responses to the user's questions about their code.
    """
    
    def __init__(self, analyzer=None):
        """
        Initialize the code mentor with an optional analyzer instance.
        
        Args:
            analyzer: The RepositoryAnalyzer instance to use for code analysis
        """
        self.analyzer = analyzer
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_analysis: Optional[Dict[str, Any]] = None
        self.repository_context: Dict[str, Any] = {} # Store detailed repo context
        self._learning_context: Dict[str, Any] = {
            'explored_files': [],
            'generated_exercises': [],
            'understanding_level': 'beginner',
            'topics_covered': [],
            'session_progress': 0
        }
        
        # Attempt to restore context from session state
        self.restore_context_from_session()

    def set_analysis_context(self, analysis_results: Dict[str, Any]):
        """Enhanced context setting with deep repository knowledge and persistence"""
        self.current_analysis = analysis_results
        
        # Build comprehensive repository context
        if analysis_results and 'analysis' in analysis_results:
            analysis = analysis_results['analysis']
            
            # Extract key files and their actual content
            detailed_files = analysis.get('detailed_files', {})
            
            # Identify the most important files for context
            key_files = self._identify_key_files(detailed_files)
            
            # Build repository-specific knowledge base
            self.repository_context = {
                'repository_name': analysis_results.get('repository', 'Unknown'),
                'primary_files': key_files,
                'architecture_summary': self._analyze_architecture(detailed_files),
                'function_inventory': self._build_function_inventory(detailed_files),
                'technology_stack': analysis.get('technologies', []),
                'quality_insights': analysis.get('quality_metrics', {}),
                'file_relationships': self._map_file_relationships(detailed_files),
                'context_timestamp': analysis_results.get('timestamp', None)
            }
            
            # Store in session state for persistence across interactions
            import streamlit as st
            if 'mentor_repository_context' not in st.session_state:
                st.session_state.mentor_repository_context = {}
            
            st.session_state.mentor_repository_context = self.repository_context.copy()
    
    def restore_context_from_session(self):
        """Restore repository context from session state if available"""
        import streamlit as st
        if hasattr(st, 'session_state') and 'mentor_repository_context' in st.session_state:
            stored_context = st.session_state.mentor_repository_context
            if stored_context and stored_context.get('repository_name'):
                self.repository_context = stored_context
                return True
        return False
    
    def _identify_key_files(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """Identify the most important files for context building"""
        key_files = {}
        
        # Prioritize by importance and content richness
        for file_path, file_info in detailed_files.items():
            importance_score = 0
            
            # High importance indicators
            file_name = Path(file_path).name.lower()
            if file_name in ['main.py', 'app.py', 'train.py', 'model.py', 'server.py']:
                importance_score += 100
            
            # Content richness indicators
            functions = file_info.get('functions', [])
            classes = file_info.get('classes', [])
            lines = file_info.get('lines', 0)
            
            importance_score += len(functions) * 10
            importance_score += len(classes) * 15
            importance_score += min(lines // 10, 50)  # Cap line bonus
            
            if importance_score > 50:  # Threshold for inclusion
                key_files[file_path] = {
                    'info': file_info,
                    'importance': importance_score,
                    'summary': self._summarize_file_purpose(file_path, file_info)
                }
        
        # Return top 8 most important files
        return dict(sorted(key_files.items(), 
                          key=lambda x: x[1]['importance'], 
                          reverse=True)[:8])

    def _summarize_file_purpose(self, file_path: str, file_info: Dict[str, Any]) -> str:
        """Generates a brief summary of a file's purpose."""
        # This is a placeholder. A real implementation would use NLP.
        if "model" in file_path: return "Core data models or ML model definitions."
        if "view" in file_path or "template" in file_path: return "UI components or page templates."
        if "controller" in file_path or "handler" in file_path: return "Request handling and business logic."
        if "main" in file_path or "app" in file_path: return "Application entry point and core logic."
        return "A module contributing to the project's functionality."

    def _analyze_architecture(self, detailed_files: Dict[str, Any]) -> str:
        """Analyze the actual architecture based on file structure and imports"""
        architecture_patterns = []
        
        file_keys = [k.lower() for k in detailed_files.keys()]
        functions_str = str([info.get('functions', []) for info in detailed_files.values()]).lower()

        # Detect MVC pattern
        has_models = any('model' in f for f in file_keys)
        has_views = any('view' in f or 'template' in f for f in file_keys)
        has_controllers = any('controller' in f or 'handler' in f for f in file_keys)
        
        if has_models and has_views and has_controllers:
            architecture_patterns.append("MVC (Model-View-Controller)")
        
        # Detect data processing pipeline
        has_data_input = any('load' in functions_str for f_info in detailed_files.values())
        has_processing = any('process' in functions_str for f_info in detailed_files.values())
        has_output = any('save' in functions_str or 'export' in functions_str for f_info in detailed_files.values())
        
        if has_data_input and has_processing and has_output:
            architecture_patterns.append("Data Processing Pipeline")
        
        # Detect ML training pipeline
        has_training = any('train' in f for f in file_keys)
        has_model_def = any('model' in f for f in file_keys)
        has_evaluation = any('eval' in f or 'test' in f for f in file_keys)
        
        if has_training and has_model_def:
            architecture_patterns.append("Machine Learning Training Pipeline")
        
        return " + ".join(architecture_patterns) if architecture_patterns else "Modular Script Architecture"
    
    def _build_function_inventory(self, detailed_files: Dict[str, Any]) -> Dict[str, List[str]]:
        """Builds an inventory of functions per file."""
        inventory = {}
        for file_path, info in detailed_files.items():
            inventory[file_path] = info.get('functions', [])
        return inventory

    def _map_file_relationships(self, detailed_files: Dict[str, Any]) -> Dict[str, List[str]]:
        """Creates a simple import-based relationship map."""
        relationships = {}
        for file_path, info in detailed_files.items():
            relationships[file_path] = info.get('imports', [])
        return relationships

    async def process_question(self, question: str) -> str:
        """
        Process a user question and provide an intelligent response.
        
        Args:
            question: The user's question about their code
            
        Returns:
            A helpful response to the user's question
        """
        try:
            self.conversation_history.append({'role': 'user', 'content': question})
            
            question_type = self._classify_question(question)
            
            if question_type == 'repository_specific':
                response = await self._repository_specific_handler(question)
            else:
                response = await self._general_question_handler(question)
            
            self.conversation_history.append({'role': 'assistant', 'content': response})
            return response
            
        except Exception as e:
            st.error(f"Mentor error: {e}")
            return f"I encountered an issue processing your question: {str(e)}. Please try rephrasing."
    
    def _classify_question(self, question: str) -> str:
        """
        Default to repository-specific when context exists, only use general for pure programming concepts.
        """
        q_lower = question.lower()
        
        # Check if we have repository context
        has_repo_context = bool(self.repository_context and self.repository_context.get('primary_files'))
        
        if has_repo_context:
            # Only classify as general for pure programming theory/concepts
            pure_general_patterns = [
                'what is python', 'what is oop', 'what is functional programming',
                'explain recursion', 'what are design patterns', 'what is mvc',
                'programming languages', 'computer science', 'algorithm complexity',
                'data structures theory', 'big o notation'
            ]
            
            # Simple greetings (short phrases only)
            if q_lower.strip() in ['hello', 'hi', 'hey', 'thanks', 'thank you']:
                return 'general'
            
            # Pure programming concepts
            if any(pattern in q_lower for pattern in pure_general_patterns):
                return 'general'
            
            # Everything else with repository context is repository-specific
            return 'repository_specific'
        
        # Without repository context, use general
        return 'general'
    
    async def _general_question_handler(self, question: str) -> str:
        """
        Handle general programming questions using AI model.
        """
        system_prompt = """You are an expert programming mentor with years of experience teaching developers. 

Your teaching philosophy:
- Guide students to understand concepts, don't just give answers
- Provide clear explanations with practical examples
- Focus on best practices and real-world applications
- Encourage learning through hands-on practice
- Break down complex topics into digestible parts

When answering programming questions:
1. Start with a clear, conceptual explanation
2. Provide a simple code example if relevant
3. Mention common pitfalls or best practices
4. Suggest next steps for deeper learning

Be encouraging, educational, and practical in your responses."""
        
        if self.analyzer and hasattr(self.analyzer, '_call_llm_hybrid'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            response = await self.analyzer._call_llm_hybrid(messages, "general_mentor_question")
            return response
        
        return "I'm unable to access the AI model right now. Please check your configuration and try again."

    async def _repository_specific_handler(self, question: str) -> str:
        """Handle questions with deep repository knowledge and comprehensive context"""
        if not self.repository_context:
            return await self._general_question_handler(question)
        
        # Build comprehensive context from actual repository analysis
        repo_name = self.repository_context.get('repository_name', 'Unknown')
        architecture = self.repository_context.get('architecture_summary', 'N/A')
        tech_stack = ', '.join(self.repository_context.get('technology_stack', []))
        
        # Enhanced repository intelligence
        repo_intelligence = self._build_enhanced_repository_context()
        
        # Include actual file contents and structure with enhanced analysis
        key_files_context = ""
        for file_path, file_data in self.repository_context.get('primary_files', {}).items():
            file_info = file_data.get('info', {})
            key_files_context += f"""
📁 FILE: {file_path}
   Purpose: {file_data.get('summary', 'N/A')}
   Key Functions: {', '.join(file_info.get('functions', [])[:5])}
   Key Classes: {', '.join(file_info.get('classes', [])[:3])}
   Size: {file_info.get('lines', 0)} lines
   Code Preview: {file_info.get('content_preview', '')[:400]}
   
   🔗 Relationships:
   - Imports: {', '.join(file_info.get('imports', [])[:5])}
   - Dependencies: {self._analyze_file_dependencies(file_path, file_info)}
---
"""
        
        comprehensive_context = f"""
You are an expert code mentor analyzing the "{repo_name}" repository. This is the CURRENT CODEBASE you are helping with.

🏗️ CURRENT REPOSITORY ANALYSIS: {repo_name}
=======================================

📊 PROJECT INTELLIGENCE:
{repo_intelligence['project_analysis']}

🏛️ ARCHITECTURE & DESIGN:
{repo_intelligence['architecture_analysis']}

🔧 TECHNOLOGY ECOSYSTEM:
{repo_intelligence['technology_analysis']}

📁 KEY FILES & THEIR ROLES:
{key_files_context}

🔄 SYSTEM INTERACTIONS:
{repo_intelligence['interaction_analysis']}

📈 QUALITY & COMPLEXITY CONTEXT:
{repo_intelligence['quality_context']}

🎯 DOMAIN-SPECIFIC PATTERNS:
{repo_intelligence['domain_patterns']}

CONTEXT INSTRUCTIONS:
- You are analyzing the "{repo_name}" codebase specifically
- All questions should be answered in the context of THIS repository
- Reference actual files, functions, and code patterns from the analysis above
- When explaining workflows or processes, refer to how they work in THIS codebase
- Assume the user is asking about THIS repository unless explicitly stated otherwise
- Use specific examples from the analyzed code to illustrate your explanations
- Connect your explanations to the actual architecture and patterns found in THIS codebase

The user's question is about the "{repo_name}" repository. Provide insights that demonstrate deep understanding of:
1. How THIS codebase's components work together
2. The specific domain/purpose of THIS project  
3. Architectural decisions made in THIS repository
4. Code quality and design patterns used in THIS codebase
5. Cross-file relationships and dependencies in THIS system

Always frame your response as being about THIS SPECIFIC CODEBASE.
"""
        
        if self.analyzer and hasattr(self.analyzer, '_call_llm_hybrid'):
            messages = [
                {"role": "system", "content": comprehensive_context},
                {"role": "user", "content": f"About the {repo_name} repository: {question}"}
            ]
            
            response = await self.analyzer._call_llm_hybrid(messages, "repository-specific analysis")
            return response
        
        return f"I'm unable to access the AI model right now for analyzing {repo_name}. Please check your configuration and try again."

    def _format_function_inventory(self) -> str:
        """Format the function inventory for context"""
        inventory = self.repository_context.get('function_inventory', {})
        formatted = ""
        
        for file_path, functions in inventory.items():
            if functions:
                formatted += f"{file_path}: {', '.join(functions[:5])}\n"
        
        return formatted[:1000]  # Limit size for context window


    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the full conversation history.
        """
        return self.conversation_history
    
    @property
    def learning_context(self) -> Dict[str, Any]:
        """
        Get the current learning context and progress.
        """
        return self._learning_context
    
    def update_learning_progress(self, file_explored: str = None, exercise_generated: str = None, 
                                topic_covered: str = None, understanding_level: str = None):
        """
        Update the learning progress tracking.
        
        Args:
            file_explored: Name of file that was explored
            exercise_generated: Description of exercise that was generated
            topic_covered: Topic that was covered in learning
            understanding_level: Current understanding level (beginner, intermediate, advanced)
        """
        if file_explored and file_explored not in self._learning_context['explored_files']:
            self._learning_context['explored_files'].append(file_explored)
        
        if exercise_generated and exercise_generated not in self._learning_context['generated_exercises']:
            self._learning_context['generated_exercises'].append(exercise_generated)
        
        if topic_covered and topic_covered not in self._learning_context['topics_covered']:
            self._learning_context['topics_covered'].append(topic_covered)
        
        if understanding_level:
            self._learning_context['understanding_level'] = understanding_level
        
        # Update session progress based on activities
        total_activities = (len(self._learning_context['explored_files']) + 
                          len(self._learning_context['generated_exercises']) + 
                          len(self._learning_context['topics_covered']))
        self._learning_context['session_progress'] = min(total_activities * 10, 100)
    
    def clear_conversation(self):
        """
        Clear the current conversation history.
        """
        self.conversation_history = []
        st.success("Conversation cleared.") 

    def _build_enhanced_repository_context(self) -> Dict[str, str]:
        """Build comprehensive repository intelligence for enhanced context"""
        primary_files = self.repository_context.get('primary_files', {})
        quality_insights = self.repository_context.get('quality_insights', {})
        tech_stack = self.repository_context.get('technology_stack', [])
        
        # Analyze project type and domain
        project_indicators = []
        domain_keywords = []
        
        for file_path, file_data in primary_files.items():
            file_info = file_data.get('info', {})
            content = file_info.get('content_preview', '').lower()
            functions = ' '.join(file_info.get('functions', [])).lower()
            imports = ' '.join(file_info.get('imports', [])).lower()
            
            # Domain detection
            if any(keyword in content + functions + imports for keyword in ['model', 'train', 'predict', 'neural', 'torch', 'tensorflow']):
                domain_keywords.append('Machine Learning')
            if any(keyword in content + functions + imports for keyword in ['api', 'endpoint', 'request', 'response', 'server']):
                domain_keywords.append('Web Service')
            if any(keyword in content + functions + imports for keyword in ['process', 'transform', 'parse', 'data', 'csv', 'json']):
                domain_keywords.append('Data Processing')
            if any(keyword in content + functions + imports for keyword in ['cli', 'command', 'args', 'parser']):
                domain_keywords.append('Command Line Tool')
        
        # Architectural pattern detection
        architecture_patterns = []
        file_names = [file_path.lower() for file_path in primary_files.keys()]
        
        if any('model' in name for name in file_names):
            architecture_patterns.append('Model-based Architecture')
        if any('controller' in name or 'handler' in name for name in file_names):
            architecture_patterns.append('Controller/Handler Pattern')
        if any('service' in name for name in file_names):
            architecture_patterns.append('Service Layer Architecture')
        if any('config' in name or 'settings' in name for name in file_names):
            architecture_patterns.append('Configuration-driven Design')
        
        # Interaction analysis
        import_frequency = {}
        for file_data in primary_files.values():
            file_info = file_data.get('info', {})
            for imp in file_info.get('imports', []):
                import_frequency[imp] = import_frequency.get(imp, 0) + 1
        
        common_dependencies = sorted(import_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'project_analysis': f"""
Project Type: {', '.join(set(domain_keywords)) if domain_keywords else 'General Purpose Application'}
Primary Domain: {max(set(domain_keywords), key=domain_keywords.count) if domain_keywords else 'Multi-purpose'}
Complexity Level: {self._assess_complexity_level()}
""",
            'architecture_analysis': f"""
Detected Patterns: {', '.join(architecture_patterns) if architecture_patterns else 'No clear patterns'}
File Organization: {self._analyze_file_organization()}
Component Structure: {self._analyze_component_structure()}
""",
            'technology_analysis': f"""
Primary Stack: {tech_stack[0] if tech_stack else 'Unknown'}
Supporting Technologies: {', '.join(tech_stack[1:4]) if len(tech_stack) > 1 else 'None detected'}
Framework Dependencies: {self._identify_frameworks()}
""",
            'interaction_analysis': f"""
Most Common Dependencies: {', '.join([f"{dep} ({count} files)" for dep, count in common_dependencies[:3]])}
Coupling Level: {self._assess_coupling_level()}
""",
            'quality_context': f"""
Code Quality Score: {quality_insights.get('overall_score', {}).get('score', 'N/A')}/100
Average Complexity: {quality_insights.get('complexity', {}).get('average_complexity', 'N/A')}
Comment Coverage: {quality_insights.get('comment_ratio', 'N/A')}%
""",
            'domain_patterns': f"""
Identified Patterns: {', '.join(set(domain_keywords)) if domain_keywords else 'General programming patterns'}
Specialized Functions: {self._identify_specialized_functions()}
"""
        }
    
    def _analyze_file_dependencies(self, file_path: str, file_info: Dict[str, Any]) -> str:
        """Analyze dependencies for a specific file"""
        imports = file_info.get('imports', [])
        if not imports:
            return "Self-contained"
        
        internal_deps = []
        external_deps = []
        
        for imp in imports:
            if any(char in imp for char in ['.', '/']):
                internal_deps.append(imp)
            else:
                external_deps.append(imp)
        
        result = []
        if internal_deps:
            result.append(f"Internal: {', '.join(internal_deps[:3])}")
        if external_deps:
            result.append(f"External: {', '.join(external_deps[:3])}")
        
        return '; '.join(result) if result else "None detected"
    
    def _assess_complexity_level(self) -> str:
        """Assess overall project complexity"""
        primary_files = self.repository_context.get('primary_files', {})
        
        total_functions = sum(len(file_data.get('info', {}).get('functions', [])) for file_data in primary_files.values())
        avg_functions = total_functions / len(primary_files) if primary_files else 0
        
        if avg_functions > 10:
            return "High - many functions per file"
        elif avg_functions > 5:
            return "Moderate - balanced structure"
        else:
            return "Low - simple organization"
    
    def _analyze_file_organization(self) -> str:
        """Analyze how files are organized"""
        primary_files = self.repository_context.get('primary_files', {})
        file_names = [file_path.lower() for file_path in primary_files.keys()]
        
        if any('main' in name or 'app' in name for name in file_names):
            return "Entry-point driven with clear main files"
        elif len(file_names) > 5:
            return "Multi-module architecture"
        else:
            return "Simple file structure"
    
    def _analyze_component_structure(self) -> str:
        """Analyze component relationships"""
        primary_files = self.repository_context.get('primary_files', {})
        
        total_classes = sum(len(file_data.get('info', {}).get('classes', [])) for file_data in primary_files.values())
        total_functions = sum(len(file_data.get('info', {}).get('functions', [])) for file_data in primary_files.values())
        
        if total_classes > total_functions * 0.3:
            return "Object-oriented with significant class usage"
        elif total_functions > total_classes * 3:
            return "Function-based with minimal classes"
        else:
            return "Mixed procedural and object-oriented approach"
    
    def _identify_frameworks(self) -> str:
        """Identify framework dependencies"""
        tech_stack = self.repository_context.get('technology_stack', [])
        primary_files = self.repository_context.get('primary_files', {})
        
        frameworks = []
        for file_data in primary_files.values():
            file_info = file_data.get('info', {})
            imports = ' '.join(file_info.get('imports', [])).lower()
            
            if 'flask' in imports:
                frameworks.append('Flask')
            if 'django' in imports:
                frameworks.append('Django')
            if 'streamlit' in imports:
                frameworks.append('Streamlit')
            if 'fastapi' in imports:
                frameworks.append('FastAPI')
            if 'torch' in imports:
                frameworks.append('PyTorch')
            if 'tensorflow' in imports:
                frameworks.append('TensorFlow')
        
        return ', '.join(set(frameworks)) if frameworks else 'No major frameworks detected'
    
    def _assess_coupling_level(self) -> str:
        """Assess coupling between components"""
        primary_files = self.repository_context.get('primary_files', {})
        
        total_imports = sum(len(file_data.get('info', {}).get('imports', [])) for file_data in primary_files.values())
        avg_imports = total_imports / len(primary_files) if primary_files else 0
        
        if avg_imports > 8:
            return "High coupling - many dependencies"
        elif avg_imports > 4:
            return "Moderate coupling - balanced dependencies"
        else:
            return "Low coupling - minimal dependencies"
    
    def _identify_specialized_functions(self) -> str:
        """Identify domain-specific function patterns"""
        primary_files = self.repository_context.get('primary_files', {})
        
        specialized_patterns = []
        all_functions = []
        
        for file_data in primary_files.values():
            file_info = file_data.get('info', {})
            all_functions.extend(file_info.get('functions', []))
        
        function_text = ' '.join(all_functions).lower()
        
        if 'analyze' in function_text or 'process' in function_text:
            specialized_patterns.append('Data Analysis')
        if 'predict' in function_text or 'model' in function_text:
            specialized_patterns.append('Machine Learning')
        if 'api' in function_text or 'endpoint' in function_text:
            specialized_patterns.append('API Development')
        if 'parse' in function_text or 'load' in function_text:
            specialized_patterns.append('Data I/O')
        
        return ', '.join(specialized_patterns) if specialized_patterns else 'General utility functions' 