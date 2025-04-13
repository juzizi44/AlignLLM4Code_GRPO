from openai_client import UserPrompts


efficiency = '''
# **Efficiency Criteria**  

**Time Complexity Optimization (Strict Evaluation)**  
- Algorithm Efficiency: Evaluate if the code utilizes **optimal algorithms** that minimize computational complexity. Any inefficient use of nested loops or suboptimal solutions must be flagged.  
- Algorithm Adaptability: Assess if the **chosen algorithm** is suitable for the application, particularly in handling large datasets or high concurrency without significant performance degradation.  
- Redundant Computation: Ensure the code does **not perform unnecessary repeated calculations**. If redundant calculations are found, it negatively impacts the evaluation score.  
- Loop Optimization: Check whether loops are **optimized** to prevent unnecessary complexity, such as excessive nesting or redundant calculations within loops.  

**Space Complexity Optimization (Strict Evaluation)**  
- Data Structure Choice: Assess the **efficiency** of the data structures used in the code. If inefficient structures are chosen, leading to higher memory consumption, the code will be rated lower.  
- Variable and Object Management: Ensure that **variables and objects** are managed efficiently. Excessive or unnecessary memory allocation due to redundant variables or objects should be flagged.  
- Caching and Reuse: Evaluate if the code utilizes **caching or object reuse** effectively to reduce redundant computations or avoid creating unnecessary objects. Poor caching strategies impact both time and space efficiency.  

**Code Optimization Practices (Strict Evaluation)**  
- Parallel and Asynchronous Optimization: Check if the code uses **parallel computing or asynchronous programming** to optimize performance for concurrent tasks. Failure to do so, when applicable, reduces the efficiency score.  
- I/O and Database Optimization: Assess if the code **minimizes and optimizes I/O operations and database queries**. Excessive or inefficient I/O and database access reduce the overall performance score.  
- Code Redundancy: Identify and flag **unnecessary code** that does not contribute to the functionality. Redundant or unused code negatively impacts performance, maintainability, and resource consumption.

# Task  
Your task is to generate 10 test cases for a given code problem based on Efficiency Criteria, to test the Efficiency of the solution.  
## Input format   
[code problem]  
## Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
  "test_case1": {{
    "input": "",
    "expected_output": "",
    "test_aspect": ""
  }},
  "test_case2": {{
    "input": "",
    "expected_output": "",
    "test_aspect": ""
  }},
...
  "test_case10": {{
    "input": "",
    "expected_output": "",
    "test_aspect": ""
  }}
}}


# Annotation  
## Code problem
{code_problem} 
### Output  
'''






robustness = '''
# Task  
Please analyze the various edge cases in the code problem and generate 10 test cases. These test cases should comprehensively cover and verify all the edge cases to ensure the robustness of the solution. Note: Please focus only on boundary cases.
## Input format   
[code problem]  
## Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
  "test_case1": {{
    "input": "",
    "expected_output": "",
    "test_aspect": ""
  }},
  "test_case2": {{
    "input": "",
    "expected_output": "",
    "test_aspect": ""
  }},
...
  "test_case10": {{
    "input": "",
    "expected_output": "",
    "test_aspect": ""
  }}
}}


# Annotation  
## Code problem
{code_problem} 
### Output  
'''

functionality = '''
# Task  
Please analyze all the requirements stated in the code problem and generate 10 test cases. These test cases should comprehensively cover and verify all the requirements mentioned in the problem statement to ensure the functional correctness of the solution.
## code problem
{code_problem}  

## Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
  "test_case1": {{
    "input": "",
    "expected_output": "",
    "test_aspect": ""
  }},
  "test_case2": {{
    "input": "",
    "expected_output": "",
    "test_aspect": ""
  }},
...
  "test_case10": {{
    "input": "",
    "expected_output": "",
    "test_aspect": ""
  }}
}}

# Output 
'''



USER_PROMPTS = UserPrompts(
    efficiency=efficiency,
    robustness=robustness,
    functionality=functionality,

)
