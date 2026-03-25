"""
Reasoning-focused LLMs

Models optimized for logical reasoning, mathematics, and complex problem-solving.

Techniques:
- Chain-of-Thought (CoT) prompting
- Self-consistency
- Tree of Thoughts
- Reasoning-enhanced training

Models:
- OpenAI o1
- DeepSeek-R1
- Reasoning-tuned open models
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ReasoningStrategy(Enum):
    """Reasoning strategies."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    SELF_CONSISTENCY = "self_consistency"
    LEAST_TO_MOST = "least_to_most"


@dataclass
class ReasoningModelInfo:
    """Information about reasoning-focused models."""
    name: str
    base_model: str
    parameters: str
    specialization: List[str]
    hf_id: Optional[str]
    capabilities: List[str]
    reasoning_method: str


class ReasoningModels:
    """
    Registry for reasoning-focused LLMs.
    """
    
    # Specialized Reasoning Models
    SPECIALIZED = {
        "deepseek-r1-7b": ReasoningModelInfo(
            name="DeepSeek-R1 7B",
            base_model="DeepSeek",
            parameters="7B",
            specialization=["Math", "Logic", "Code reasoning"],
            hf_id="deepseek-ai/DeepSeek-R1-7B",
            capabilities=["Step-by-step reasoning", "Mathematical proofs"],
            reasoning_method="Reinforcement Learning from Reasoning",
        ),
        "wizardmath-7b": ReasoningModelInfo(
            name="WizardMath 7B",
            base_model="Mistral",
            parameters="7B",
            specialization=["Mathematics", "Problem solving"],
            hf_id="WizardLM/WizardMath-7B-V1.1",
            capabilities=["Math word problems", "Equation solving"],
            reasoning_method="Evol-Instruct for Math",
        ),
        "wizardmath-13b": ReasoningModelInfo(
            name="WizardMath 13B",
            base_model="LLaMA 2",
            parameters="13B",
            specialization=["Advanced mathematics"],
            hf_id="WizardLM/WizardMath-13B-V1.0",
            capabilities=["Complex math", "Multi-step reasoning"],
            reasoning_method="Evol-Instruct for Math",
        ),
        "metamath-7b": ReasoningModelInfo(
            name="MetaMath 7B",
            base_model="LLaMA 2",
            parameters="7B",
            specialization=["Mathematical reasoning"],
            hf_id="meta-math/MetaMath-7B-V1.0",
            capabilities=["GSM8K", "MATH dataset"],
            reasoning_method="MetaMath augmentation",
        ),
    }
    
    # Proprietary Reasoning Models
    PROPRIETARY = {
        "openai-o1": ReasoningModelInfo(
            name="OpenAI o1",
            base_model="GPT-4 family",
            parameters="Unknown",
            specialization=["Complex reasoning", "STEM", "Coding"],
            hf_id=None,
            capabilities=["Extended thinking", "PhD-level reasoning"],
            reasoning_method="Reinforcement learning with chain-of-thought",
        ),
        "openai-o1-mini": ReasoningModelInfo(
            name="OpenAI o1-mini",
            base_model="GPT-4 family",
            parameters="Unknown",
            specialization=["STEM", "Coding"],
            hf_id=None,
            capabilities=["Fast reasoning", "Cost-efficient"],
            reasoning_method="Lightweight reasoning model",
        ),
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, ReasoningModelInfo]:
        """Get all reasoning models."""
        all_models = {}
        all_models.update(cls.SPECIALIZED)
        all_models.update(cls.PROPRIETARY)
        return all_models


class ReasoningPrompts:
    """
    Collection of reasoning prompt templates.
    """
    
    @staticmethod
    def chain_of_thought(problem: str) -> str:
        """Chain-of-thought prompting."""
        return f"""Let's solve this step by step:

Problem: {problem}

Solution:
Let's think through this carefully:
1."""
    
    @staticmethod
    def step_by_step(problem: str) -> str:
        """Step-by-step reasoning prompt."""
        return f"""Solve this problem step by step:

{problem}

Step 1:"""
    
    @staticmethod
    def zero_shot_cot(problem: str) -> str:
        """Zero-shot chain-of-thought."""
        return f"""{problem}

Let's think step by step:"""
    
    @staticmethod
    def few_shot_cot(problem: str, examples: List[Dict[str, str]]) -> str:
        """Few-shot chain-of-thought with examples."""
        prompt = "Here are some examples:\n\n"
        
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Problem: {ex['problem']}\n"
            prompt += f"Solution: {ex['solution']}\n\n"
        
        prompt += f"Now solve this problem:\n{problem}\n\nSolution:"
        return prompt
    
    @staticmethod
    def self_consistency_prompt(problem: str, n: int = 5) -> str:
        """Prompt for self-consistency (generate multiple reasoning paths)."""
        return f"""Solve this problem with detailed reasoning:

{problem}

Reasoning path:"""
    
    @staticmethod
    def least_to_most(problem: str) -> str:
        """Least-to-most prompting (break down complex problems)."""
        return f"""Let's break down this problem into smaller sub-problems and solve them one by one:

Main problem: {problem}

Sub-problems:
1."""
    
    @staticmethod
    def verify_reasoning(problem: str, solution: str) -> str:
        """Verification prompt."""
        return f"""Problem: {problem}

Proposed solution: {solution}

Please verify this solution step by step. Is it correct? If not, what's wrong?

Verification:"""


class ReasoningEngine:
    """
    Engine for enhanced reasoning with LLMs.
    """
    
    def __init__(self, model, strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT):
        """
        Initialize reasoning engine.
        
        Args:
            model: Base LLM model (must have generate() method)
            strategy: Reasoning strategy to use
        """
        self.model = model
        self.strategy = strategy
    
    def solve_problem(
        self,
        problem: str,
        strategy: Optional[ReasoningStrategy] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Solve a problem using the specified reasoning strategy.
        
        Args:
            problem: Problem to solve
            strategy: Override default strategy
            examples: Few-shot examples (if applicable)
            temperature: Sampling temperature
        """
        strategy = strategy or self.strategy
        
        # Generate prompt based on strategy
        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            prompt = ReasoningPrompts.chain_of_thought(problem)
        elif strategy == ReasoningStrategy.STEP_BY_STEP:
            prompt = ReasoningPrompts.step_by_step(problem)
        elif strategy == ReasoningStrategy.LEAST_TO_MOST:
            prompt = ReasoningPrompts.least_to_most(problem)
        else:
            prompt = ReasoningPrompts.zero_shot_cot(problem)
        
        # Add examples for few-shot
        if examples:
            prompt = ReasoningPrompts.few_shot_cot(problem, examples)
        
        # Generate solution
        solution = self.model.generate(
            prompt,
            temperature=temperature,
            max_length=1024,
            **kwargs
        )
        
        if isinstance(solution, list):
            solution = solution[0]
        
        return solution
    
    def solve_with_self_consistency(
        self,
        problem: str,
        n_samples: int = 5,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[str, Dict[str, int]]:
        """
        Solve problem using self-consistency (multiple reasoning paths).
        
        Returns:
            Most consistent answer and vote counts
        """
        prompt = ReasoningPrompts.self_consistency_prompt(problem)
        
        # Generate multiple solutions
        solutions = []
        for _ in range(n_samples):
            solution = self.model.generate(
                prompt,
                temperature=temperature,
                max_length=1024,
                **kwargs
            )
            if isinstance(solution, list):
                solution = solution[0]
            solutions.append(solution)
        
        # Extract final answers (simplified - looks for numbers or last sentence)
        answers = [self._extract_answer(sol) for sol in solutions]
        
        # Vote for most common answer
        from collections import Counter
        vote_counts = Counter(answers)
        most_common = vote_counts.most_common(1)[0][0]
        
        return most_common, dict(vote_counts)
    
    def _extract_answer(self, solution: str) -> str:
        """Extract final answer from solution (simplified)."""
        # Look for patterns like "Therefore, the answer is X"
        keywords = ["answer is", "final answer", "therefore", "thus"]
        
        lines = solution.split("\n")
        for line in reversed(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                return line.strip()
        
        # Fallback: return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return solution.strip()
    
    def verify_solution(
        self,
        problem: str,
        solution: str,
        **kwargs
    ) -> Dict[str, any]:
        """
        Verify a proposed solution.
        
        Returns:
            Dict with verification result and feedback
        """
        prompt = ReasoningPrompts.verify_reasoning(problem, solution)
        
        verification = self.model.generate(
            prompt,
            temperature=0.3,  # Lower temp for verification
            max_length=512,
            **kwargs
        )
        
        if isinstance(verification, list):
            verification = verification[0]
        
        # Simple heuristic to determine if correct
        verification_lower = verification.lower()
        is_correct = "correct" in verification_lower and "not correct" not in verification_lower
        
        return {
            "is_correct": is_correct,
            "verification": verification,
        }


class MathSolver:
    """
    Specialized solver for mathematical problems.
    """
    
    def __init__(self, model):
        self.model = model
        self.reasoning_engine = ReasoningEngine(model)
    
    def solve_math_problem(
        self,
        problem: str,
        show_work: bool = True,
        verify: bool = False,
        **kwargs
    ) -> Dict[str, any]:
        """
        Solve a mathematical problem.
        
        Args:
            problem: Math problem description
            show_work: Include step-by-step solution
            verify: Verify the solution
        """
        # Enhance prompt for math
        math_prompt = f"""Solve this math problem step by step. Show all work and explain each step.

Problem: {problem}

Solution:
"""
        
        solution = self.model.generate(
            math_prompt,
            temperature=0.3,  # Lower temp for math
            max_length=1024,
            **kwargs
        )
        
        if isinstance(solution, list):
            solution = solution[0]
        
        result = {
            "problem": problem,
            "solution": solution,
        }
        
        # Extract numerical answer
        answer = self._extract_numerical_answer(solution)
        if answer:
            result["answer"] = answer
        
        # Verify if requested
        if verify:
            verification = self.reasoning_engine.verify_solution(problem, solution)
            result["verification"] = verification
        
        return result
    
    def _extract_numerical_answer(self, solution: str) -> Optional[float]:
        """Extract numerical answer from solution."""
        import re
        
        # Look for patterns like "= 42" or "answer is 42"
        patterns = [
            r'=\s*([-+]?\d+\.?\d*)',
            r'answer is\s*([-+]?\d+\.?\d*)',
            r'result is\s*([-+]?\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, solution.lower())
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    continue
        
        return None


if __name__ == "__main__":
    print("=== Reasoning-focused LLMs ===\n")
    
    all_models = ReasoningModels.get_all_models()
    for key, model in all_models.items():
        print(f"{key}:")
        print(f"  {model.name} ({model.parameters})")
        print(f"  Specialization: {', '.join(model.specialization)}")
        print(f"  Method: {model.reasoning_method}")
        print()
    
    print("\n=== Reasoning Strategies ===")
    for strategy in ReasoningStrategy:
        print(f"- {strategy.value}")
    
    print("\n=== Usage Example ===")
    print("""
# Load a reasoning model
from architectures.decoder_only import DecoderOnlyModel
model = DecoderOnlyModel.from_pretrained("WizardLM/WizardMath-7B-V1.1")

# Create reasoning engine
engine = ReasoningEngine(model, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)

# Solve a problem
problem = "If a train travels 120 km in 2 hours, what is its average speed?"
solution = engine.solve_problem(problem)

# Use self-consistency for more reliable answers
answer, votes = engine.solve_with_self_consistency(problem, n_samples=5)

# Math solver
math_solver = MathSolver(model)
result = math_solver.solve_math_problem(
    "What is the sum of the first 10 prime numbers?",
    verify=True
)
    """)
