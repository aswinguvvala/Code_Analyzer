# code_mentor.py
import streamlit as st
from typing import Dict, List, Any, Optional
import json
import re
from pathlib import Path

class InteractiveCodeMentor:
    """
    An AI-powered mentor that understands the codebase and can teach developers
    how it works through interactive conversations and exercises.
    """
    
    def __init__(self, analysis_data: Dict[str, Any], llm_service):
        self.analysis_data = analysis_data
        self.llm_service = llm_service  # Your hybrid LLM service
        self.conversation_history = []
        self.learning_context = {
            'current_topic': None,
            'explored_files': [],
            'generated_exercises': [],
            'understanding_level': 'beginner'
        }
    
    def start_mentor_session(self) -> Dict[str, Any]:
        """Initialize a mentoring session with welcome and capabilities overview."""
        welcome_message = """
        👋 Hello! I'm your Code Mentor for this repository.
        
        I can help you understand:
        - How specific features work
        - The architecture and design patterns
        - Why certain decisions were made
        - How to contribute to this codebase
        
        You can ask me questions like:
        - "How does authentication work?"
        - "Explain the data flow in this app"
        - "Show me how to add a new feature"
        - "What does the UserController do?"
        
        I can also create exercises to help you learn!
        """
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': welcome_message
        })
        
        return {
            'message': welcome_message,
            'suggestions': self._generate_initial_suggestions()
        }
    
    def _generate_initial_suggestions(self) -> List[str]:
        """Generate smart suggestions based on the codebase analysis."""
        suggestions = []
        
        # Suggest based on main technologies
        tech = self.analysis_data.get('technologies', [])
        if 'Python' in tech and 'Flask' in str(self.analysis_data):
            suggestions.append("How does the Flask routing work in this app?")
        elif 'React' in str(self.analysis_data):
            suggestions.append("Explain the component hierarchy")
        
        # Suggest based on entry points
        entry_points = self.analysis_data.get('analysis', {}).get('quality_metrics', {}).get(
            'visual_analysis', {}).get('entry_points', [])
        if entry_points:
            main_entry = entry_points[0]
            suggestions.append(f"Walk me through what happens when {main_entry['name']} runs")
        
        # Suggest based on complexity
        complex_functions = self._find_complex_functions()
        if complex_functions:
            suggestions.append(f"Explain the {complex_functions[0]['name']} function")
        
        # General suggestions
        suggestions.extend([
            "What's the overall architecture of this project?",
            "Show me the main data flow",
            "Create an exercise for understanding the core functionality"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _find_complex_functions(self) -> List[Dict[str, Any]]:
        """Find complex functions that might need explanation."""
        complex_functions = []
        
        quality_metrics = self.analysis_data.get('analysis', {}).get('quality_metrics', {})
        complexity_data = quality_metrics.get('complexity', {}).get('functions', [])
        
        for func in complexity_data:
            if func.get('complexity', 0) > 10:
                complex_functions.append(func)
        
        return sorted(complex_functions, key=lambda x: x.get('complexity', 0), reverse=True)
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a user question and generate an intelligent response.
        This is where the magic happens - understanding intent and providing value.
        """
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': question
        })
        
        # Determine question type and intent
        question_analysis = self._analyze_question_intent(question)
        
        # Route to appropriate handler
        if question_analysis['type'] == 'how_does_work':
            response = await self._explain_how_something_works(
                question_analysis['topic'], question
            )
        elif question_analysis['type'] == 'explain_code':
            response = await self._explain_specific_code(
                question_analysis['topic'], question
            )
        elif question_analysis['type'] == 'architecture':
            response = await self._explain_architecture(question)
        elif question_analysis['type'] == 'create_exercise':
            response = await self._create_learning_exercise(
                question_analysis['topic'], question
            )
        elif question_analysis['type'] == 'data_flow':
            response = await self._trace_data_flow(
                question_analysis['topic'], question
            )
        else:
            response = await self._general_question_handler(question)
        
        # Add response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response['message']
        })
        
        # Update learning context
        self._update_learning_context(question_analysis['topic'])
        
        return response
    
    def _analyze_question_intent(self, question: str) -> Dict[str, str]:
        """
        Analyze the user's question to understand what they're asking about.
        This uses pattern matching and keyword analysis.
        """
        question_lower = question.lower()
        
        # Patterns for different question types
        patterns = {
            'how_does_work': [
                r'how does (.*) work',
                r'how do (.*) work',
                r'explain how (.*)',
                r'walk me through (.*)',
                r'what happens when (.*)'
            ],
            'explain_code': [
                r'explain the (.*) function',
                r'what does (.*) do',
                r'explain (.*) class',
                r'show me (.*) code'
            ],
            'architecture': [
                r'architecture',
                r'overall structure',
                r'design pattern',
                r'how is .* organized'
            ],
            'create_exercise': [
                r'create an exercise',
                r'give me a task',
                r'practice',
                r'challenge'
            ],
            'data_flow': [
                r'data flow',
                r'trace the data',
                r'flow of data',
                r'data pipeline'
            ]
        }
        
        # Check each pattern type
        for intent_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, question_lower)
                if match:
                    topic = match.group(1) if match.groups() else None
                    return {
                        'type': intent_type,
                        'topic': topic,
                        'original_question': question
                    }
        
        # Default to general question
        return {
            'type': 'general',
            'topic': None,
            'original_question': question
        }
    
    async def _explain_how_something_works(self, topic: str, full_question: str) -> Dict[str, Any]:
        """
        Explain how a specific feature or component works.
        This creates a comprehensive explanation with examples.
        """
        # Find relevant files and code
        relevant_context = self._find_relevant_code_context(topic)
        
        # Build context for LLM
        context = f"""
        Repository Analysis:
        {json.dumps(self.analysis_data.get('analysis', {}).get('file_structure', {}), indent=2)}
        
        Relevant Files:
        {json.dumps(relevant_context['files'], indent=2)}
        
        File Explanations:
        {json.dumps(relevant_context['explanations'], indent=2)}
        
        Visual Analysis:
        {json.dumps(relevant_context['visual_data'], indent=2)}
        """
        
        # Generate explanation
        messages = [
            {
                "role": "system",
                "content": """You are a patient coding mentor explaining how code works.
                Create a clear, step-by-step explanation that a developer can follow.
                Use examples from the actual code, create simple diagrams, and explain
                the 'why' behind design decisions."""
            },
            {
                "role": "user",
                "content": f"""
                Question: {full_question}
                Topic: {topic}
                
                Context:
                {context}
                
                Please provide:
                1. A high-level overview of how {topic} works
                2. Step-by-step breakdown of the implementation
                3. Key files and functions involved
                4. A simple diagram showing the flow
                5. Common gotchas or important notes
                
                Make it educational and easy to understand.
                """
            }
        ]
        
        llm_response = await self.llm_service(messages, "code_explanation")
        
        # Generate a visual diagram if appropriate
        diagram = self._generate_explanation_diagram(topic, relevant_context)
        
        # Find code examples
        code_examples = self._extract_code_examples(topic, relevant_context)
        
        return {
            'message': llm_response,
            'diagram': diagram,
            'code_examples': code_examples,
            'related_files': relevant_context['files'][:5],
            'follow_up_suggestions': self._generate_follow_up_questions(topic)
        }
    
    def _find_relevant_code_context(self, topic: str) -> Dict[str, Any]:
        """Find files and code relevant to the topic being asked about."""
        relevant_context = {
            'files': [],
            'explanations': {},
            'visual_data': {}
        }
        
        topic_lower = topic.lower() if topic else ""
        
        # Search through file analysis
        file_explanations = self.analysis_data.get('analysis', {}).get('file_explanations', {})
        
        for file_path, explanation in file_explanations.items():
            # Score relevance based on multiple factors
            relevance_score = 0
            
            # Check if topic appears in file name
            if topic_lower in file_path.lower():
                relevance_score += 3
            
            # Check if topic appears in explanation
            if topic_lower in explanation.lower():
                relevance_score += 2
            
            # Check for related terms
            related_terms = self._get_related_terms(topic_lower)
            for term in related_terms:
                if term in explanation.lower():
                    relevance_score += 1
            
            if relevance_score > 0:
                relevant_context['files'].append({
                    'path': file_path,
                    'score': relevance_score
                })
                relevant_context['explanations'][file_path] = explanation
        
        # Sort by relevance
        relevant_context['files'].sort(key=lambda x: x['score'], reverse=True)
        
        # Add visual analysis data if available
        visual_data = self.analysis_data.get('analysis', {}).get('quality_metrics', {}).get(
            'visual_analysis', {})
        if visual_data:
            relevant_context['visual_data'] = {
                'entry_points': visual_data.get('entry_points', [])[:3],
                'execution_flows': visual_data.get('execution_flows', [])[:2]
            }
        
        return relevant_context
    
    def _get_related_terms(self, topic: str) -> List[str]:
        """Get terms related to the topic for better context finding."""
        # This is a simple implementation - could be enhanced with word embeddings
        related_terms_map = {
            'auth': ['authentication', 'login', 'user', 'session', 'token', 'security'],
            'database': ['db', 'sql', 'model', 'schema', 'query', 'orm'],
            'api': ['endpoint', 'route', 'rest', 'request', 'response'],
            'ui': ['component', 'view', 'render', 'display', 'interface'],
            'test': ['testing', 'unit', 'integration', 'mock', 'assert']
        }
        
        related = []
        for key, terms in related_terms_map.items():
            if key in topic:
                related.extend(terms)
        
        return related
    
    def _generate_explanation_diagram(self, topic: str, context: Dict[str, Any]) -> Optional[str]:
        """Generate a Mermaid diagram to visualize the explanation."""
        # This creates a simple flow diagram based on the topic
        if not context['files']:
            return None
        
        diagram_lines = ["graph TD"]
        
        # Add nodes for relevant files
        for i, file_info in enumerate(context['files'][:5]):
            file_name = Path(file_info['path']).stem
            node_id = f"F{i}"
            diagram_lines.append(f'    {node_id}["{file_name}"]')
        
        # Add connections based on imports or relationships
        # This is simplified - in production, analyze actual imports
        if len(context['files']) > 1:
            for i in range(len(context['files'][:5]) - 1):
                diagram_lines.append(f'    F{i} --> F{i+1}')
        
        # Add topic node
        diagram_lines.append(f'    Topic["{topic}"]')
        diagram_lines.append(f'    Topic --> F0')
        
        return "\n".join(diagram_lines)
    
    def _extract_code_examples(self, topic: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract relevant code examples for the topic."""
        examples = []
        
        # Get detailed file information
        detailed_files = self.analysis_data.get('analysis', {}).get('detailed_files', {})
        
        for file_info in context['files'][:3]:
            file_path = file_info['path']
            if file_path in detailed_files:
                file_data = detailed_files[file_path]
                
                # Find relevant functions
                functions = file_data.get('functions', [])
                for func in functions[:2]:
                    if topic.lower() in func.lower():
                        examples.append({
                            'type': 'function',
                            'name': func,
                            'file': file_path,
                            'description': f"Function {func} from {Path(file_path).name}"
                        })
        
        return examples
    
    def _generate_follow_up_questions(self, topic: str) -> List[str]:
        """Generate intelligent follow-up questions based on the current topic."""
        questions = []
        
        # Generic follow-ups
        questions.extend([
            f"Can you show me an example of using {topic}?",
            f"What are the common mistakes when working with {topic}?",
            f"How would I modify {topic} for a new feature?"
        ])
        
        # Add context-specific questions
        if 'auth' in topic.lower():
            questions.append("How is user session managed?")
        elif 'database' in topic.lower():
            questions.append("What's the schema design philosophy?")
        elif 'api' in topic.lower():
            questions.append("How is API versioning handled?")
        
        return questions[:3]
    
    async def _create_learning_exercise(self, topic: str, full_question: str) -> Dict[str, Any]:
        """
        Create an interactive learning exercise based on the codebase.
        This is where we become a true mentor - creating hands-on learning.
        """
        # Determine exercise difficulty based on user's understanding level
        difficulty = self._determine_exercise_difficulty()
        
        # Find a suitable part of the codebase for the exercise
        exercise_context = self._find_exercise_context(topic, difficulty)
        
        # Generate the exercise using LLM
        messages = [
            {
                "role": "system",
                "content": """You are a coding mentor creating educational exercises.
                Create exercises that are practical, relevant to the actual codebase,
                and help developers understand how things work by doing."""
            },
            {
                "role": "user",
                "content": f"""
                Create a {difficulty} level coding exercise about: {topic or 'the core functionality'}
                
                Codebase context:
                {json.dumps(exercise_context, indent=2)}
                
                The exercise should include:
                1. Clear objective (what they'll learn)
                2. Starting code or setup instructions
                3. Specific task to complete
                4. Hints (without giving away the solution)
                5. Success criteria (how they know they've done it right)
                6. Bonus challenges for advanced learners
                
                Make it practical and tied to the actual codebase.
                """
            }
        ]
        
        exercise_details = await self.llm_service(messages, "exercise_generation")
        
        # Create exercise structure
        exercise = {
            'id': f"exercise_{len(self.learning_context['generated_exercises'])}",
            'topic': topic or 'general',
            'difficulty': difficulty,
            'content': exercise_details,
            'related_files': exercise_context['files'],
            'hints_available': True,
            'solution_available': False  # Unlocked after attempt
        }
        
        # Store exercise
        self.learning_context['generated_exercises'].append(exercise)
        
        return {
            'message': exercise_details,
            'exercise': exercise,
            'setup_commands': self._generate_setup_commands(exercise_context),
            'resources': self._gather_learning_resources(topic)
        }
    
    def _determine_exercise_difficulty(self) -> str:
        """Determine appropriate exercise difficulty based on user's progress."""
        explored_files = len(self.learning_context['explored_files'])
        completed_exercises = len([e for e in self.learning_context['generated_exercises'] 
                                  if e.get('completed', False)])
        
        if explored_files < 3 and completed_exercises == 0:
            return 'beginner'
        elif explored_files < 10 and completed_exercises < 3:
            return 'intermediate'
        else:
            return 'advanced'
    
    def _find_exercise_context(self, topic: Optional[str], difficulty: str) -> Dict[str, Any]:
        """Find appropriate code context for creating an exercise."""
        context = {
            'files': [],
            'functions': [],
            'existing_tests': []
        }
        
        # Get detailed files
        detailed_files = self.analysis_data.get('analysis', {}).get('detailed_files', {})
        
        # For beginner exercises, find simple files
        if difficulty == 'beginner':
            for file_path, file_data in detailed_files.items():
                if file_data.get('lines', 0) < 100:  # Small files
                    context['files'].append(file_path)
                    context['functions'].extend(file_data.get('functions', [])[:2])
        
        # For intermediate, find files with moderate complexity
        elif difficulty == 'intermediate':
            quality_metrics = self.analysis_data.get('analysis', {}).get('quality_metrics', {})
            complexity_data = quality_metrics.get('complexity', {}).get('functions', [])
            
            for func in complexity_data:
                if 5 <= func.get('complexity', 0) <= 10:
                    context['functions'].append(func)
        
        # For advanced, find complex areas that could be refactored
        else:
            refactoring_opportunities = self.analysis_data.get('analysis', {}).get(
                'quality_metrics', {}).get('overall_score', {}).get('recommendations', [])
            context['refactoring_opportunities'] = refactoring_opportunities
        
        return context
    
    def _generate_setup_commands(self, context: Dict[str, Any]) -> List[str]:
        """Generate commands to set up the exercise environment."""
        commands = []
        
        # Basic setup
        commands.append("# Clone the repository if you haven't already")
        commands.append("git clone <repository-url>")
        commands.append("cd <repository-name>")
        
        # Create exercise branch
        commands.append("\n# Create a branch for your exercise")
        commands.append("git checkout -b learning-exercise")
        
        # File-specific setup
        if context.get('files'):
            commands.append(f"\n# You'll be working with: {context['files'][0]}")
            commands.append(f"open {context['files'][0]}")
        
        return commands
    
    def _gather_learning_resources(self, topic: Optional[str]) -> List[Dict[str, str]]:
        """Gather relevant learning resources based on the topic."""
        resources = []
        
        # Add documentation files from the repository
        key_files = self.analysis_data.get('analysis', {}).get('key_files', [])
        for file in key_files:
            if 'readme' in file.lower() or 'doc' in file.lower():
                resources.append({
                    'type': 'documentation',
                    'title': f"Project {file}",
                    'location': file
                })
        
        # Add relevant code files
        if topic:
            relevant_context = self._find_relevant_code_context(topic)
            for file_info in relevant_context['files'][:3]:
                resources.append({
                    'type': 'code',
                    'title': f"Implementation: {Path(file_info['path']).name}",
                    'location': file_info['path']
                })
        
        return resources
    
    def _update_learning_context(self, topic: Optional[str]):
        """Update the learning context based on user interactions."""
        if topic:
            self.learning_context['current_topic'] = topic
        
        # Track explored files
        for msg in self.conversation_history[-2:]:  # Last exchange
            content = msg.get('content', '')
            # Simple file detection - could be enhanced
            if '.py' in content or '.js' in content:
                # Extract file names mentioned
                import re
                files = re.findall(r'[\w/]+\.\w+', content)
                self.learning_context['explored_files'].extend(files)
        
        # Remove duplicates
        self.learning_context['explored_files'] = list(set(
            self.learning_context['explored_files']
        ))
    
    async def provide_hint(self, exercise_id: str, hint_level: int = 1) -> str:
        """Provide progressive hints for an exercise."""
        exercise = next((e for e in self.learning_context['generated_exercises'] 
                        if e['id'] == exercise_id), None)
        
        if not exercise:
            return "Exercise not found."
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful mentor providing hints without giving away the solution."
            },
            {
                "role": "user",
                "content": f"""
                Exercise: {exercise['content']}
                
                Provide a level {hint_level} hint:
                - Level 1: General direction
                - Level 2: More specific guidance
                - Level 3: Almost the solution but they still need to write code
                
                Make the hint helpful but still educational.
                """
            }
        ]
        
        hint = await self.llm_service(messages, "hint_generation")
        return hint
    
    async def check_solution(self, exercise_id: str, solution_code: str) -> Dict[str, Any]:
        """Check a user's solution and provide feedback."""
        exercise = next((e for e in self.learning_context['generated_exercises'] 
                        if e['id'] == exercise_id), None)
        
        if not exercise:
            return {"success": False, "message": "Exercise not found."}
        
        messages = [
            {
                "role": "system",
                "content": """You are a coding mentor reviewing a student's solution.
                Provide constructive feedback that helps them learn."""
            },
            {
                "role": "user",
                "content": f"""
                Exercise: {exercise['content']}
                
                Student's solution:
                ```
                {solution_code}
                ```
                
                Please:
                1. Check if the solution meets the exercise requirements
                2. Identify what they did well
                3. Suggest improvements (if any)
                4. Explain any concepts they might have missed
                5. Provide a score (0-100)
                
                Be encouraging and educational.
                """
            }
        ]
        
        feedback = await self.llm_service(messages, "solution_review")
        
        # Mark exercise as completed if score is good
        # This is simplified - you'd want to parse the LLM response properly
        if "score" in feedback.lower() and any(str(n) in feedback for n in range(70, 101)):
            exercise['completed'] = True
            self.learning_context['understanding_level'] = 'intermediate'
        
        return {
            "success": True,
            "feedback": feedback,
            "exercise_completed": exercise.get('completed', False),
            "next_steps": self._suggest_next_learning_steps(exercise)
        }
    
    def _suggest_next_learning_steps(self, completed_exercise: Dict[str, Any]) -> List[str]:
        """Suggest what to learn next based on completed exercise."""
        suggestions = []
        
        if completed_exercise['difficulty'] == 'beginner':
            suggestions.append("Try an intermediate exercise on the same topic")
            suggestions.append("Explore how this feature integrates with others")
        elif completed_exercise['difficulty'] == 'intermediate':
            suggestions.append("Take on an advanced refactoring challenge")
            suggestions.append("Learn about the architectural decisions behind this code")
        else:
            suggestions.append("Consider contributing to the actual project!")
            suggestions.append("Mentor others on what you've learned")
        
        return suggestions