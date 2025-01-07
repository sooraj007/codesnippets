from smolagents import CodeAgent, tool, LiteLLMModel
from typing import Dict, Any
import sys
import subprocess
import tempfile
import os
import json
import re

class CodeDebugger:
    MULTI_LANGUAGE_REGEX_GRAMMAR = {
        "type": "regex",
        "value": r"Thought: .+?\nCode:\n``````(?:\n)?"
    }

    @staticmethod
    def parse_code_block(code_blob: str) -> str:
        """Parse code from markdown code blocks with flexible language support"""
        patterns = [
            r"Thought: .+?\nCode:\n``````",
            r"``````",
            r"``````"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, code_blob, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        raise ValueError(f"No code block found in response:\n{code_blob}")

    @staticmethod
    @tool
    def execute_code(code: str) -> Dict[str, Any]:
        """Execute the provided code and return the result or error.
        
        Args:
            code: The JavaScript code to execute
            
        Returns:
            A dictionary containing execution results with keys:
            - success: Whether execution was successful
            - output: Program output if successful
            - error: Error message if unsuccessful
        """
        try:
            # Parse the code block if it's in markdown format
            if "```" in code:
                code = CodeDebugger.parse_code_block(code)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            try:
                result = subprocess.run(
                    ['node', temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                os.unlink(temp_file_path)

                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr if result.returncode != 0 else None
                }

            except subprocess.TimeoutExpired:
                os.unlink(temp_file_path)
                return {
                    "success": False,
                    "output": None,
                    "error": "Code execution timed out"
                }

        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }

class CodeAnalyzerAgent(CodeAgent):
    def __init__(self, tools, model, **kwargs):
        super().__init__(
            tools=tools,
            model=model,
            grammar=CodeDebugger.MULTI_LANGUAGE_REGEX_GRAMMAR,
            **kwargs
        )

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Comprehensive code analysis with execution and static analysis"""
        execution_result = CodeDebugger.execute_code(code)
        
        analysis_result = self.run(f"""
        Analyze this code comprehensively:
        {code}

        Execution Result:
        Success: {execution_result['success']}
        Output: {execution_result['output']}
        Error: {execution_result['error']}

        Provide a detailed analysis focusing on:
        1. Syntax issues
        2. Potential runtime errors
        3. Performance bottlenecks
        4. Best practice violations
        5. Suggested improvements

        Return a structured dictionary with these keys:
        {{
            "has_issues": bool,
            "issues": list,
            "suggestions": list,
            "complexity_score": float
        }}
        """)

        try:
            # Ensure the result is a valid dictionary
            if isinstance(analysis_result, str):
                analysis_result = json.loads(analysis_result.strip('```json\n'))
            
            return {
                "has_issues": analysis_result.get('has_issues', bool(execution_result['error'])),
                "issues": analysis_result.get('issues', [execution_result['error']] if execution_result['error'] else []),
                "suggestions": analysis_result.get('suggestions', []),
                "complexity_score": analysis_result.get('complexity_score', 0.0)
            }
        except Exception:
            return {
                "has_issues": bool(execution_result['error']),
                "issues": [execution_result['error']] if execution_result['error'] else [],
                "suggestions": [],
                "complexity_score": 0.0
            }

class CodeFixerAgent(CodeAgent):
    def __init__(self, tools, model, **kwargs):
        super().__init__(
            tools=tools,
            model=model,
            grammar=CodeDebugger.MULTI_LANGUAGE_REGEX_GRAMMAR,
            **kwargs
        )

    def fix_code(self, code: str, analysis: Dict[str, Any]) -> str:
        """Advanced code fixing with multiple strategies"""
        max_attempts = 5
        current_code = code

        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1} to fix the code")
            
            fix_result = self.run(f"""
            Fix the code with these constraints:
            - Current Code: {current_code}
            - Issues: {analysis['issues']}
            - Suggestions: {analysis['suggestions']}

            Provide:
            1. Completely refactored code
            2. Explanation of changes
            3. Performance and readability improvements

            Return ONLY the fixed code, without markdown or explanations.
            """)

            # Clean and validate the fixed code
            fixed_code = fix_result.strip()
            if fixed_code.startswith("```"):
                fixed_code = "\n".join(fixed_code.split("\n")[1:-1])

            execution_result = CodeDebugger.execute_code(fixed_code)
            
            if execution_result['success']:
                print(f"Code fixed successfully! Output:\n{execution_result['output']}")
                return fixed_code

            # Update analysis for next iteration
            analysis = {
                "has_issues": True,
                "issues": [f"Runtime error: {execution_result['error']}"],
                "suggestions": ["Fix the runtime error"],
                "complexity_score": 0.0
            }
            current_code = fixed_code

        return current_code

def create_debugging_agents(model_name: str = "ollama/qwen2.5-coder:14b"):
    """Create and return debugging agents with a specified model"""
    model = LiteLLMModel(model_id=model_name)
    
    analyzer = CodeAnalyzerAgent(
        tools=[CodeDebugger.execute_code],
        model=model,
        add_base_tools=True,
        planning_interval=2,
        max_steps=5
    )
    
    fixer = CodeFixerAgent(
        tools=[CodeDebugger.execute_code],
        model=model,
        add_base_tools=True,
        max_steps=5
    )
    
    return analyzer, fixer

def debug_code(code_file: str, model_name: str = "ollama/qwen2.5-coder:14b") -> str:
    """Main function to debug and fix code"""
    try:
        with open(code_file, 'r') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file {code_file}: {str(e)}")
        return None

    analyzer, fixer = create_debugging_agents(model_name)

    # Comprehensive code analysis
    analysis_result = analyzer.analyze_code(code)

    # Fix code if issues are found
    if analysis_result["has_issues"]:
        print("Issues found:", analysis_result["issues"])
        print("Suggested fixes:", analysis_result["suggestions"])
        
        fixed_code = fixer.fix_code(code, analysis_result)
        
        # Write fixed code
        try:
            with open("debug.js", "w") as f:
                f.write(fixed_code.strip())
            print("\nFixed code has been written to debug.js")
        except Exception as e:
            print(f"Error writing to debug.js: {str(e)}")
        
        return fixed_code

    print("No issues found in the code.")
    return code

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a file to debug")
        print("Usage: python debugger_agent.py <file>")
        sys.exit(1)

    code_file = sys.argv[1]
    fixed_code = debug_code(code_file)
    
    if fixed_code:
        print("\nFixed code:")
        print(fixed_code)
