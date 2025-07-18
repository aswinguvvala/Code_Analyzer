import asyncio
import streamlit as st
from typing import Dict, List, Any, Optional

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
        # Initialize learning context
        self._learning_context = {
            'explored_files': [],
            'generated_exercises': [],
            'understanding_level': 'beginner'
        }
    
    @property
    def learning_context(self) -> Dict[str, Any]:
        """
        Get the current learning context with progress information.
        
        Returns:
            Dictionary containing learning progress data
        """
        return self._learning_context
    
    def update_learning_progress(self, files_explored: List[str] = None, exercises: List[str] = None, level: str = None):
        """
        Update the learning progress context.
        
        Args:
            files_explored: List of files that have been explored
            exercises: List of exercises that have been generated
            level: Current understanding level
        """
        if files_explored:
            self._learning_context['explored_files'] = files_explored
        if exercises:
            self._learning_context['generated_exercises'] = exercises
        if level:
            self._learning_context['understanding_level'] = level

    def set_analysis_context(self, analysis_results: Dict[str, Any]):
        """
        Set the current analysis context for the mentor to reference.
        
        Args:
            analysis_results: The results from repository analysis
        """
        self.current_analysis = analysis_results
        # Update learning context with explored files from analysis
        if analysis_results and 'analysis' in analysis_results:
            file_structure = analysis_results['analysis'].get('file_structure', {})
            files = file_structure.get('files', [])
            self.update_learning_progress(files_explored=files)
    
    async def process_question(self, question: str) -> str:
        """
        Process a user question and provide an intelligent response.
        
        Args:
            question: The user's question about their code
            
        Returns:
            A helpful response to the user's question
        """
        try:
            # Add the question to conversation history
            self.conversation_history.append({
                'type': 'question',
                'content': question,
                'timestamp': 'now'
            })
            
            # Determine the type of question and route appropriately
            question_type = self._classify_question(question)
            
            if question_type == 'analysis_specific':
                response = await self._analysis_specific_handler(question)
            elif question_type == 'code_explanation':
                response = await self._code_explanation_handler(question)
            elif question_type == 'improvement_suggestions':
                response = await self._improvement_suggestions_handler(question)
            else:
                # This is the method that was missing - now implemented
                response = await self._general_question_handler(question)
            
            # Add response to conversation history
            self.conversation_history.append({
                'type': 'response',
                'content': response,
                'timestamp': 'now'
            })
            
            return response
            
        except Exception as e:
            # Provide a helpful fallback response instead of crashing
            return f"I encountered an issue processing your question: {str(e)}. Could you try rephrasing your question or asking something more specific about your code analysis?"
    
    def _classify_question(self, question: str) -> str:
        """
        Classify the type of question to determine how to handle it.
        
        Args:
            question: The user's question
            
        Returns:
            The classification of the question type
        """
        question_lower = question.lower()
        
        # Look for keywords that indicate different types of questions
        if any(word in question_lower for word in ['analysis', 'result', 'score', 'quality']):
            return 'analysis_specific'
        elif any(word in question_lower for word in ['function', 'class', 'code', 'how does', 'what does']):
            return 'code_explanation'
        elif any(word in question_lower for word in ['improve', 'better', 'fix', 'optimize', 'recommend']):
            return 'improvement_suggestions'
        else:
            return 'general'
    
    async def _general_question_handler(self, question: str) -> str:
        """
        Handle general questions about code analysis and programming concepts.
        This is the method that was missing and causing the error.
        
        Args:
            question: The user's general question
            
        Returns:
            A helpful response to the general question
        """
        try:
            # If we have an analyzer available, use it to generate a response
            if self.analyzer and hasattr(self.analyzer, '_call_llm_hybrid'):
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a helpful code mentor. Answer the user's question about programming, code analysis, or software development in a clear and educational way."
                    },
                    {
                        "role": "user", 
                        "content": f"User question: {question}\n\nPlease provide a helpful and educational response."
                    }
                ]
                
                response = await self.analyzer._call_llm_hybrid(messages, "general question")
                return response
            else:
                # Fallback response when no analyzer is available
                return self._provide_fallback_response(question)
                
        except Exception:
            return f"I'm having trouble accessing my AI capabilities right now. Here's what I can tell you: {self._provide_fallback_response(question)}"
    
    async def _analysis_specific_handler(self, question: str) -> str:
        """Handle questions specific to the current analysis results."""
        if not self.current_analysis:
            return "I don't have any analysis results to reference. Please run an analysis first, then ask your question."
        
        # Extract relevant information from the current analysis
        analysis = self.current_analysis.get('analysis', {})
        quality_metrics = analysis.get('quality_metrics', {})
        
        # Build context about the current analysis
        context = f"""
        Current Analysis Context:
        - Repository: {self.current_analysis.get('repository', 'Unknown')}
        - Total Files: {analysis.get('file_structure', {}).get('total_files', 'Unknown')}
        - Quality Score: {quality_metrics.get('overall_score', {}).get('score', 'Unknown')}
        - Technologies: {', '.join(analysis.get('technologies', []))}
        """
        
        if self.analyzer and hasattr(self.analyzer, '_call_llm_hybrid'):
            messages = [
                {
                    "role": "system", 
                    "content": f"You are a code mentor helping interpret analysis results. {context}"
                },
                {
                    "role": "user", 
                    "content": question
                }
            ]
            
            response = await self.analyzer._call_llm_hybrid(messages, "analysis question")
            return response
        else:
            return f"Based on your analysis results: {context}\n\nI'd be happy to help, but I need more specific information about what aspect of the analysis you'd like to understand better."
    
    async def _code_explanation_handler(self, question: str) -> str:
        """Handle questions about specific code explanations."""
        return await self._general_question_handler(f"Please explain this code-related question: {question}")
    
    async def _improvement_suggestions_handler(self, question: str) -> str:
        """Handle questions about code improvements."""
        context = ""
        if self.current_analysis:
            quality_metrics = self.current_analysis.get('analysis', {}).get('quality_metrics', {})
            recommendations = quality_metrics.get('overall_score', {}).get('recommendations', [])
            context = f"Current recommendations: {'; '.join(recommendations)}"
        
        return await self._general_question_handler(f"Considering code improvement: {question}. {context}")
    
    def _provide_fallback_response(self, question: str) -> str:
        """
        Provide a helpful fallback response when AI services aren't available.
        
        Args:
            question: The user's question
            
        Returns:
            A basic but helpful response
        """
        question_lower = question.lower()
        
        if 'quality' in question_lower:
            return """Code quality typically involves several factors:
            - Readability: Is the code easy to understand?
            - Maintainability: Can the code be easily modified?
            - Complexity: Are functions and classes reasonably sized?
            - Documentation: Are there helpful comments and documentation?
            - Testing: Does the code have adequate test coverage?"""
        
        elif 'improve' in question_lower:
            return """Common ways to improve code include:
            - Breaking large functions into smaller, focused functions
            - Adding clear variable and function names
            - Including helpful comments for complex logic
            - Removing duplicate code
            - Following consistent coding style
            - Adding error handling"""
        
        elif 'analysis' in question_lower:
            return """Code analysis helps you understand:
            - The structure and organization of your codebase
            - Potential areas for improvement
            - Code quality metrics and scores
            - Dependencies between different parts of your code
            - Best practices compliance"""
        
        else:
            return """I'm here to help you understand your code analysis results and improve your coding practices. 
            You can ask me about:
            - Code quality metrics and what they mean
            - How to improve specific aspects of your code
            - Understanding the analysis results
            - Programming best practices
            
            Feel free to ask me a more specific question!"""
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history for display."""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = [] 