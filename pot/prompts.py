"""Coding, continuation and public-error feedback templates.

Template literal hashes are recorded in docs/original-prompt-provenance.json.
Benchmark interfaces and code-selection instructions are added by pot.search."""

import re

PROMPT_VERSION = "coding-templates-v1"


def render_template(template: str, **values: str) -> str:
    # Replace named slots once; preserve literal Python braces in old examples.
    return re.sub(r"\{([a-z_]+)\}", lambda m: values.get(m[1], m[0]), template)

CODING_START_PROMPT = '''
Solve this coding problem step by step with both algorithmic thinking and Python implementation.

Problem: {problem}

Instructions:
1. Analyze the problem and explain your approach clearly
2. For each reasoning step, provide the corresponding Python code in  ```python ... ``` tags
3. Ensure each code snippet is executable and testable
4. Build your solution incrementally - each step should be runnable
5. When you reach a complete solution, format it as:  ```python ... ```

Example format:
Step 1: I need to understand the input format and constraints.
```python
def solve_problem(nums, target):
    # Initialize variables
    result = []
    return result
```

Step 2: Implement the core algorithm logic.
```python
def solve_problem(nums, target):
    # Use hash map for O(n) solution
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

Begin your solution:
'''

CODING_CONTINUE_PROMPT = '''
Continue solving this coding problem with both algorithmic thinking and Python implementation.

Problem: {problem}

Previous work:
{previous_steps}

Instructions:
1. Review what has been done so far - does it align with the problem requirements?
2. Continue with the next logical step using executable Python code
3. Provide Python code in ```python ... ``` tags for each reasoning step
4. CRITICAL: Ensure your code is complete and runnable at each step
5. Check if your current approach will lead to a valid solution
6. When complete, output: ```python ... ```

REFLECTION QUESTIONS to consider:
- Does my current approach address the original problem statement?
- Are my code implementations correct and executable?
- Am I making progress toward a working solution?
- Have I missed any constraints or edge cases from the problem?

Continue with the next step:
'''

CODE_ERROR_FEEDBACK_PROMPT = '''
Your previous code had execution issues. Here's the feedback:

Error Type: {error_type}
Error Message: {error_message}
Pass Rate: {pass_rate}
Previous Code: 
{code}

Failed Test Cases:
{failed_cases}

Please:
1. Analyze the error and understand what went wrong
2. Identify the root cause of the issue
3. Provide a corrected version of the code
4. Ensure the fix addresses the specific error type

Remember to:
- Check syntax and logic carefully
- Handle edge cases properly
- Ensure your code matches the expected input/output format
- Test your logic mentally before implementing

Continue with the corrected approach:
'''

REFLECTION_PROMPT = '''
Before proceeding, let's reflect on the current solution approach:

Problem: {problem}
Current Progress: {current_steps}

CRITICAL REFLECTION:
1. Am I solving the correct problem as stated?
2. Are my algorithmic steps logically sound and efficient?
3. Do my code implementations correctly represent my reasoning?
4. Am I making measurable progress toward a working solution?
5. Have I considered all constraints and edge cases?

Based on this reflection:
- Should I continue with the current approach?
- Do I need to backtrack and try a different algorithm?
- Are there any bugs or logical errors I need to fix?

Provide your reflection and then continue or adjust your approach accordingly.
'''
