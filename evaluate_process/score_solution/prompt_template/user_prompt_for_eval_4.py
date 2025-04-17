comment = '''
# Code Commenting Evaluation Criteria (Total Score: 30 Points)

Comments should accurately convey the purpose and logic of the code, ensuring the information is concise, easy to understand, and unambiguous. They should follow a consistent style and standard, aligning with the overall code structure while explaining key parts. Over-commenting or omission should be avoided to ensure appropriate and effective information delivery.

## Comment Readability (9 Points)

Comments should be concise, clear, and accurately worded to effectively aid in understanding the code logic.

### Language Clarity (3 Points)

- 3 Points: Comments are concise and clear, with no redundant or ambiguous expressions, and the language is fluent and smooth.
- 2 Points: Some comments contain minor redundancy or slight ambiguity but do not affect overall comprehension.
- 1 Point: Many comments are ambiguous or overly verbose, affecting readability.
- 0 Points: Most comments are difficult to understand and require extra effort to interpret.

### Terminology Usage (3 Points)

- 3 Points: Terms are accurate, all technical terms are appropriately explained, and align with the code logic.
- 2 Points: Terminology is mostly accurate, but some lack necessary explanations.
- 1 Point: Incorrect or inconsistent terminology may cause misunderstandings.
- 0 Points: Terms are used arbitrarily, making comprehension difficult or misleading developers.

### Complex Logic Background Information (3 Points)

- 3 Points: All complex algorithms or business logic are well-explained with background information.
- 2 Points: Most complex logic has background information, but some areas need further clarification.
- 1 Point: Only some key logic has background explanations, requiring additional effort to understand.
- 0 Points: Completely lacks background information, making complex logic difficult to understand.

## Comment Completeness (9 Points)

Comments should fully describe the functionality, key logic, and edge cases of the code, ensuring developers can quickly understand its purpose and implementation.

### Function Description (3 Points)

- 3 Points: Comments clearly and completely explain the function and purpose of the code block, making it understandable without reading the code.
- 2 Points: The function description is relatively clear but needs some improvements.
- 1 Point: Only some code blocks have function descriptions, lacking overall consistency.
- 0 Points: No function descriptions are provided.

### Key Logic and Algorithm Explanation (3 Points)

- 3 Points: All complex algorithms and key logic are well-commented, including algorithm concepts and key steps, facilitating understanding and maintenance.
- 2 Points: Basic explanations are provided for algorithms or logic, but some critical points lack details.
- 1 Point: Only input and output are described, without explaining the algorithm process or implementation idea.
- 0 Points: No explanations for algorithms at all.

### Edge Cases and Exception Handling (3 Points)

- 3 Points: Clearly describes edge cases, exception handling logic, and special condition handling.
- 2 Points: Some edge cases or exceptions are commented on, but gaps remain.
- 1 Point: Comments exist only for specific situations, ignoring some potential issues.
- 0 Points: No comments on edge cases or exceptions.

## Comment Consistency (6 Points)

Comments should follow a uniform format and language style to avoid inconsistencies that impact readability.

### Formatting Standards (3 Points)

- 3 Points: Strictly follows project or industry standards (e.g., Javadoc, Python Docstring, Doxygen), maintaining a uniform format and best practices.
- 2 Points: Mostly follows formatting standards, but some inconsistencies exist.
- 1 Point: Formatting is inconsistent, mixing multiple styles, which affects readability.
- 0 Points: Formatting is chaotic, non-standard, and severely impacts readability.

### Language Consistency (3 Points)

- 3 Points: Comments are entirely in English and maintain a consistent language style.
- 2 Points: Very few instances of mixed-language comments, but they do not impact overall understanding.
- 1 Point: Mixed Chinese and English comments, affecting readability.
- 0 Points: Random language switching in comments, severely impacting comprehension.

## Appropriate Commenting (6 Points)

Comments should be reasonably distributed, providing useful information without excessive redundancy that hinders code readability.

### Comment Density (3 Points)

- 3 Points: The density of comments is appropriate, matching the complexity of the code logic with sufficient information, neither excessive nor lacking.
- 2 Points: Some code sections have too many or too few comments, but overall comprehension is still possible.
- 1 Point: A large number of unnecessary comments or too few comments affect readability.
- 0 Points: Comments are extremely sparse or entirely absent.

### Distracting Comments (3 Points)

- 3 Points: No redundant, outdated, or repetitive comments; all comments are meaningful and effective.
- 2 Points: A few redundant or unnecessary comments exist, but they have minimal impact.
- 1 Point: A noticeable amount of outdated or repetitive comments, affecting readability.
- 0 Points: A large number of outdated or meaningless comments, severely interfering with code readability.

# Task
Your task is to rank the code comment quality of the four solutions based on the Code Comment Scoring Criteria above. Use the criteria to score the four solutions below and provide the final ranking. If there are solutions with the same score during sorting, please make your own judgment.

## Input format 
[solution1]
[solution2]
[solution3]
[solution4]

   
### Output format 
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
"solution1": {{
    "comment_readability": {{
      "total_score": ,
      "language_clarity": {{
        "score": ,
        "reason": ""
      }},
      "terminology_usage": {{
        "score": ,
        "reason": ""
      }},
      "complex_logic_background": {{
        "score": ,
        "reason": ""
      }}
    }},
    "comment_completeness": {{
      "total_score": ,
      "functional_description": {{
        "score": ,
        "reason": ""
      }},
      "key_logic_explanation": {{
        "score": ,
        "reason": ""
      }},
      "edge_cases_exception_handling": {{
        "score": ,
        "reason": ""
      }}
    }},
    "comment_consistency": {{
      "total_score": ,
      "formatting_standards": {{
        "score": ,
        "reason": ""
      }},
      "language_consistency": {{
        "score": ,
        "reason": ""
      }}
    }},
    "appropriate_commenting": {{
      "total_score": ,
      "comment_density": {{
        "score": ,
        "reason": ""
      }},
      "distracting_comments": {{
        "score": ,
        "reason": ""
      }}
    }},
    "solution_final_score": 
  }},
  "solution2": {{
    ...Same as above...
  }},
  "solution3": {{
    ...Same as above...
  }},
  "solution4": {{
    ...Same as above...
  }},
}}

# Annotation  
## Code problem
{code_problem} 
## Assistant’s Response  
### solution1  
{solution1}   
### solution2  
{solution2}   
### solution3
{solution3}
### solution4
{solution4}

### Output
'''



efficiency = '''
# Efficiency Scoring Criteria (Total 30 Points)

Code efficiency refers to the ability of the code to accomplish tasks with the least resource consumption (such as time, memory, computational power, etc.) during execution. Efficient code is capable of processing large amounts of data in a short time while ensuring stable operation with limited resources, avoiding unnecessary performance bottlenecks or wastage.

## Time Complexity Optimization (12 Points)

The code should reduce unnecessary computations, avoid inefficient algorithms, and optimize the execution speed of key logic.

### Algorithm Efficiency (3 Points)

- 3 Points: Uses optimal or near-optimal algorithms with time complexity optimized as much as possible (e.g., O(n) instead of O(n²)).
- 2 Points: The algorithm is generally reasonable but still has room for optimization (e.g., could be replaced with a more efficient algorithm).
- 1 Point: Uses suboptimal algorithms, causing performance degradation, but the impact is not critical.
- 0 Points: The algorithm is inefficient, with obvious optimization needs.

### Algorithm Adaptability (3 Points)

- 3 Points: The algorithm is highly suited to the actual application scenario, capable of efficiently handling large-scale data or high concurrency.
- 2 Points: The algorithm can generally adapt to the scenario, but efficiency is lower in some cases.
- 1 Point: The algorithm choice is not suitable for the current application scenario, limiting performance.
- 0 Points: The algorithm is completely unsuitable for the current needs, significantly impacting program performance.

### Redundant Computation (3 Points)

- 3 Points: No redundant computation, the code logic is streamlined and efficient.
- 2 Points: There is minor redundant computation, but the overall impact is minimal.
- 1 Point: Obvious redundant computation, affecting performance.
- 0 Points: There is significant redundant computation, severely slowing down execution speed.

### Loop Optimization (3 Points)

- 3 Points: Loop optimization is appropriate, avoiding unnecessary nesting and repeated calculations.
- 2 Points: The loops are generally reasonable, but there is still room for optimization in some parts.
- 1 Point: Inefficient loop writing, affecting execution speed.
- 0 Points: Large amounts of inefficient loops, severely degrading performance.

## Space Complexity Optimization (9 Points)

The code should reasonably use data structures, avoid unnecessary memory usage, and reduce temporary variables and redundant data storage.

### Data Structure Choice (3 Points)

- 3 Points: Uses the optimal data structure, reasonably matching the usage scenario.
- 2 Points: The data structure is generally suitable, but there are areas that could be optimized.
- 1 Point: Uses suboptimal data structures, affecting storage efficiency.
- 0 Points: The choice of data structure is clearly inappropriate, impacting performance.

### Variable and Object Management (3 Points)

- 3 Points: Proper use of variables and objects, avoiding unnecessary memory occupation.
- 2 Points: There are a small number of redundant variables or objects, but the impact is minimal.
- 1 Point: Multiple unnecessary variables or objects, causing resource waste.
- 0 Points: A large number of redundant variables or objects, occupying unnecessary memory resources.

### Caching and Reuse (3 Points)

- 3 Points: Fully utilizes caching and object reuse mechanisms, reducing unnecessary repeated creation.
- 2 Points: Some areas could optimize caching and reuse strategies, but the overall impact is minimal.
- 1 Point: Limited use of caching, leading to repeated calculations or unnecessary object creation.
- 0 Points: No caching used, resulting in a lot of repeated calculations or unnecessary object creation.

## Code Optimization Practices (9 Points)

The code should follow good optimization strategies, such as reducing I/O operations, optimizing database queries, and avoiding unnecessary locks, to improve overall execution efficiency.

### Parallel and Asynchronous Optimization (3 Points)

- 3 Points: Appropriately uses multithreading, parallel computing, or asynchronous programming to improve execution efficiency.
- 2 Points: Some scenarios could optimize parallel or asynchronous strategies, but the impact is minimal.
- 1 Point: Parallel or asynchronous processing has obvious optimization space, affecting code efficiency.
- 0 Points: Does not use parallel or asynchronous optimization, leading to low performance. Contains inefficient synchronous operations, severely slowing down the program's runtime.

### I/O and Database Optimization (3 Points)

- 3 Points: Reduces I/O operations and optimizes database queries to improve access efficiency.
- 2 Points: I/O or database access still has room for optimization, but overall it is acceptable.
- 1 Point: I/O or database query design is unreasonable, affecting program performance.
- 0 Points: Contains large amounts of inefficient I/O or database queries, leading to significant performance issues.

### Code Redundancy (3 Points)

- 3 Points: The code is concise, with no redundant logic, and all parts serve a practical purpose.
- 2 Points: Very few redundant parts, but the overall impact is minimal.
- 1 Point: Some unused code, affecting code maintainability and execution efficiency.
- 0 Points: A large amount of unused code, severely affecting code maintainability and execution efficiency.

# Task  
Your task is to rank the Code Efficiency of the four solutions based on the Code Efficiency Scoring Criteria above. Use the criteria to score the four solutions below and provide the final ranking. If there are solutions with the same score during sorting, please make your own judgment.  

## Input format    
[solution1]
[solution2]
[solution3]
[solution4]
 


### Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
  "solution1": {{
    "time_complexity_optimization": {{
      "total_score": ,
      "algorithm_efficiency": {{
        "score": ,
        "reason": ""
      }},
      "algorithm_adaptability": {{
        "score": ,
        "reason": ""
      }},
      "redundant_computation": {{
        "score": ,
        "reason": ""
      }},
      "loop_optimization": {{
        "score": ,
        "reason": ""
      }}
    }},
    "space_complexity_optimization": {{
      "total_score": ,
      "data_structure_choice": {{
        "score": ,
        "reason": ""
      }},
      "variable_object_management": {{
        "score": ,
        "reason": ""
      }},
      "caching_and_reuse": {{
        "score": ,
        "reason": ""
      }}
    }},
    "code_optimization_practices": {{
      "total_score": ,
      "parallel_asynchronous_optimization": {{
        "score": ,
        "reason": ""
      }},
      "io_database_optimization": {{
        "score": ,
        "reason": ""
      }},
      "code_redundancy": {{
        "score": ,
        "reason": ""
      }}
    }},
    "solution_final_score": 
  }},
  "solution2": {{
    ...Same as above...
  }},
  "solution3": {{
    ...Same as above...
  }},
  "solution4": {{
    ...Same as above...
  }},

}}



# Annotation  
## Code problem
{code_problem} 
## Assistant’s Response  
### solution1  
{solution1}   
### solution2  
{solution2}   
### solution3
{solution3}
### solution4
{solution4}
  

### Output  
'''



modularity = '''
# Modularization Scoring Criteria (Total 30 Points)

Code modularity refers to the practice of dividing code into multiple independent, reusable modules, each responsible for a specific function or task. Each module should have a clear interface and boundaries, allowing for independent development, testing, and maintenance. It also reduces the coupling between modules, enhancing the system's scalability, maintainability, and reusability.

## Code Structure Rationality (12 Points)

Module division should be clear, ensuring that each module has a single, independent responsibility, avoiding high coupling and confused responsibilities.

### Single Responsibility Principle (3 Points)

- 3 Points: Each module is responsible for a single task, following the single responsibility principle.
- 2 Points: Most modules have clear responsibilities, but some modules have multiple responsibilities.
- 1 Point: Multiple modules take on too many different responsibilities, leading to high coupling and low readability.
- 0 Points: Module responsibilities are confused, with a single module taking on multiple unrelated tasks, making the code hard to maintain and expand.

### Module Independence (3 Points)

- 3 Points: Modules are highly independent, communicating only through clear interfaces.
- 2 Points: Modules are generally independent, but there are some unnecessary dependencies.
- 1 Point: Several modules are highly coupled, lacking clear boundaries.
- 0 Points: Modules are heavily interdependent, making it difficult to test or replace them independently, resulting in high system coupling.

### Code Organization Structure (3 Points)

- 3 Points: File and directory structure is clear, following best practices, and module division is reasonable.
- 2 Points: Overall structure is good, but some modules are poorly positioned.
- 1 Point: The code organization is chaotic, making it difficult to quickly understand module relationships.
- 0 Points: The code structure is messy, with illogical module division and a lack of reasonable file and directory organization, making understanding and maintenance extremely difficult.

### Module Dependency Relationships (3 Points)

- 3 Points: Dependencies between modules are clear, with a simple dependency graph, avoiding complex dependency chains.
- 2 Points: Module dependencies are generally clear, but there are some instances of confusing dependencies or circular dependencies.
- 1 Point: Dependencies between modules are unclear, with some modules having complex dependency chains.
- 0 Points: Module dependencies are complex, with numerous circular dependencies or implicit coupling, making code modifications prone to errors.

## Code Reusability (9 Points)

Modules should be designed as reusable components, avoiding redundant code and improving development efficiency and code quality.

### Code Reusability Level (3 Points)

- 3 Points: High reusability, core functionality is well encapsulated, and code reuse rate is high.
- 2 Points: Most code is reusable, but some duplicate logic still exists.
- 1 Point: Only part of the code is reusable, with a significant amount of redundant code.
- 0 Points: No code reuse, all functionality is implemented repeatedly, leading to a large amount of redundant code.

### Common Module Encapsulation (3 Points)

- 3 Points: Common functionalities have been encapsulated into independent modules and applied appropriately.
- 2 Points: Some functionalities are well modularized, but there is still code that could be extracted.
- 1 Point: Only a few functions are encapsulated, with a large amount of code not modularized.
- 0 Points: No common module encapsulation, all functionality is written directly into business code, resulting in very poor maintainability.

### Redundant Code Elimination (3 Points)

- 3 Points: Redundant code has been avoided, and similar functionalities have been encapsulated into generic modules.
- 2 Points: Most redundant code has been eliminated, but some similar functionalities have not been extracted.
- 1 Point: Redundant code is severe, with a lack of reasonable modularization and extraction.
- 0 Points: The code is filled with a large amount of redundant logic, and similar functions are not encapsulated, making the code bloated and difficult to modify.

## Module Interface Design (9 Points)

Modules should interact through clear interfaces, and interface design should follow the principles of encapsulation and scalability.

### Interface Clarity (3 Points)

- 3 Points: Interface names are standardized, parameters are clear, documentation is complete, and the interface is easy to understand and use.
- 2 Points: The interface is generally clear, but some parameters or names are not intuitive.
- 1 Point: The interface design is not standardized, naming is chaotic, and usage is inconvenient.
- 0 Points: The interface design is chaotic, naming is not standardized, there is no documentation, and the calling method is obscure and hard to understand, making the module difficult to use.

### High Cohesion (3 Points)

- 3 Points: The internal functions of the module are highly related, with a single responsibility, tight logic, and ease of maintenance and expansion.
- 2 Points: The internal functions of the module are generally related, but there are some scattered responsibilities.
- 1 Point: The internal functions of the module are loosely related, with unclear responsibilities, making maintenance difficult.
- 0 Points: The internal logic of the module is chaotic, with unrelated functions, a lack of unity, making code maintenance extremely difficult.

### Low Coupling (3 Points)

- 3 Points: Dependencies between modules are clear, interacting through interfaces, and modifying one module has minimal impact on other modules.
- 2 Points: There are minor dependencies between modules, but the overall coupling is within an acceptable range.
- 1 Point: Modules are highly dependent on other modules, making it difficult to use or modify them independently.
- 0 Points: Modules are severely coupled, directly accessing the internal implementation of other modules, and any modification could affect the entire system, resulting in high maintenance costs. 

# Task  
Your task is to rank the Modularity of the four solutions based on the Modularity Scoring Criteria above. Use the criteria to score the four solutions below and provide the final ranking. If there are solutions with the same score during sorting, please make your own judgment.  

## Input format   
[solution1]
[solution2]
[solution3]
[solution4]
 


### Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
  "solution1": {{
    "code_structure_rationality": {{
      "total_score": ,
      "single_responsibility_principle": {{
        "score": ,
        "reason": ""
      }},
      "module_independence": {{
        "score": ,
        "reason": ""
      }},
      "code_organization_structure": {{
        "score": ,
        "reason": ""
      }},
      "module_dependency_relationships": {{
        "score": ,
        "reason": ""
      }}
    }},
    "code_reusability": {{
      "total_score": ,
      "code_reusability_level": {{
        "score": ,
        "reason": ""
      }},
      "common_module_encapsulation": {{
        "score": ,
        "reason": ""
      }},
      "redundant_code_elimination": {{
        "score": ,
        "reason": ""
      }}
    }},
    "module_interface_design": {{
      "total_score": ,
      "interface_clarity": {{
        "score": ,
        "reason": ""
      }},
      "high_cohesion": {{
        "score": ,
        "reason": ""
      }},
      "low_coupling": {{
        "score": ,
        "reason": ""
      }}
    }},
    "solution_final_score": 
  }},
  "solution2": {{
    ...Same as above...
  }},
  "solution3": {{
    ...Same as above...
  }},
  "solution4": {{
    ...Same as above...
  }},

}}



# Annotation  
## Code problem
{code_problem} 
## Assistant’s Response  
### solution1  
{solution1}   
### solution2  
{solution2}   
### solution3
{solution3}
### solution4
{solution4}
  

### Output  
'''


simplicity = '''
# Simplicity Scoring Criteria (Total 30 Points)

Code simplicity refers to implementing requirements with minimal code, efficient logical structures, and clear expressions while ensuring completeness and readability. It avoids redundancy, repetition, and unnecessary complexity, thereby enhancing maintainability and understandability.

## Code Structure Simplicity (12 Points)

The code structure should be clear and concise, avoiding unnecessary complexity and layers, making the code easy to read and understand.

### Code Depth (3 Points)

- 3 Points: The code structure is reasonable, with appropriate depth, flat structure, and easy to track.
- 2 Points: The code has multiple levels, but overall readability is acceptable with no major difficulty in understanding.
- 1 Point: The code has excessive depth, high complexity, and is difficult to understand and maintain.
- 0 Points: The code depth is excessively complex with no logical structure, making it extremely difficult to understand and maintain.

### Function/Method Length (3 Points)

- 3 Points: Functions/methods are short, with a single responsibility, and easy to understand.
- 2 Points: Function/method length is moderate, covering multiple steps but does not affect readability.
- 1 Point: Functions/methods are too long, with multiple responsibilities, making them difficult to understand and debug.
- 0 Points: Functions/methods are unusually long, difficult to trace, with unclear responsibilities, and difficult to understand.

### Code Duplication (3 Points)

- 3 Points: No code duplication, all functionalities are modularized and reused, avoiding redundancy.
- 2 Points: Most code is not duplicated, but there are some repeated logic.
- 1 Point: Code duplication is severe, requiring extensive extraction and refactoring.
- 0 Points: Excessive code duplication, making it nearly impossible to reuse and leading to redundancy and maintenance difficulty.

### Ineffective/Redundant Code (3 Points)

- 3 Points: No ineffective or redundant code, every part contributes to the implementation.
- 2 Points: Some redundant code exists, but it does not affect the code execution.
- 1 Point: There is a large amount of redundant or ineffective code, which affects code simplicity and performance.
- 0 Points: The code contains large amounts of ineffective and redundant parts, severely impacting functionality and code efficiency.

## Code Readability (12 Points)

The code should be simple, readable, with clear naming, easy to understand, and not verbose.

### Variable and Function Naming (3 Points)

- 3 Points: Naming is concise and descriptive, accurately expressing the purpose, avoiding overly long or short names.
- 2 Points: Naming is generally clear, but some names are a bit unclear or too long.
- 1 Point: Naming is not intuitive, too short or too long, hindering understanding.
- 0 Points: Naming completely fails to meet standards, making it impossible to understand its purpose or meaning.

### Code Comments (3 Points)

- 3 Points: Comments are concise and effective, clearly describing the purpose and key points of the code, avoiding redundant comments.
- 2 Points: Comments are relatively clear, but some parts have excessive or inaccurate comments.
- 1 Point: Comments are unclear, lacking necessary explanations or being too verbose.
- 0 Points: No comments or comments are completely unclear, making it difficult to understand the purpose and implementation of the code.

### Control Structure Simplicity (3 Points)

- 3 Points: Control structures are simple and clear, avoiding unnecessary nesting and complex conditions.
- 2 Points: Control structures are generally clear, but some conditional statements or nesting are excessive.
- 1 Point: Control structures are complex, with too much nesting, affecting readability.
- 0 Points: Control structures are overly complex, with chaotic logic, making them hard to understand or maintain.

### Code Style Consistency (3 Points)

- 3 Points: Code style is uniform, following consistent conventions (e.g., indentation, spaces, brackets) with no style conflicts.
- 2 Points: Code style is mostly consistent, but there are slight inconsistencies in some places.
- 1 Point: Code style is inconsistent, with noticeable deviations or incorrect conventions.
- 0 Points: Code style is severely inconsistent, with numerous convention errors, affecting code readability.

## Code Simplicity Improvement (6 Points)

Code should be further simplified through the use of specific techniques and design patterns to enhance complex function implementations.

### Use of Advanced Language Features (3 Points)

- 3 Points: Advanced language features (e.g., list comprehensions, lambda functions, generators) are used to simplify code and improve readability and simplicity.
- 2 Points: Some advanced language features are applied, but there is still room for simplification in some areas.
- 1 Point: Little or no use of advanced language features, and the code can be further simplified.
- 0 Points: No use of advanced language features, resulting in overly complex and hard-to-understand code.

### Design Patterns and Best Practices (3 Points)

- 3 Points: Appropriate design patterns (e.g., Singleton, Factory) and best practices are used effectively to reduce redundant code.
- 2 Points: Some design patterns or best practices are applied, but not all situations are optimized.
- 1 Point: Design patterns and best practices are underused, and the code can be further optimized and simplified.
- 0 Points: No design patterns or best practices are used, resulting in chaotic and hard-to-expand code. 

# Task  
Your task is to rank the Simplicity of the four solutions based on the Simplicity Scoring Criteria above. Use the criteria to score the four solutions below and provide the final ranking. If there are solutions with the same score during sorting, please make your own judgment.  

## Input format   
[solution1]
[solution2]
[solution3]
[solution4]
 
 
       

### Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
  "solution1": {{
    "code_structure_simplicity": {{
      "total_score": ,
      "code_depth": {{
        "score": ,
        "reason": ""
      }},
      "function_method_length": {{
        "score": ,
        "reason": ""
      }},
      "code_duplication": {{
        "score": ,
        "reason": ""
      }},
      "ineffective_redundant_code": {{
        "score": ,
        "reason": ""
      }}
    }},
    "code_readability": {{
      "total_score": ,
      "variable_function_naming": {{
        "score": ,
        "reason": ""
      }},
      "code_comments": {{
        "score": ,
        "reason": ""
      }},
      "control_structure_simplicity": {{
        "score": ,
        "reason": ""
      }},
      "code_style_consistency": {{
        "score": ,
        "reason": ""
      }}
    }},
    "code_simplicity_improvement": {{
      "total_score": ,
      "use_of_advanced_language_features": {{
        "score": ,
        "reason": ""
      }},
      "design_patterns_best_practices": {{
        "score": ,
        "reason": ""
      }}
    }},
    "solution_final_score": 
  }},
  "solution2": {{
    ...Same as above...
  }},
  "solution3": {{
    ...Same as above...
  }},
  "solution4": {{
    ...Same as above...
  }},

}}


# Annotation  
## Code problem
{code_problem} 
## Assistant’s Response  
### solution1  
{solution1}   
### solution2  
{solution2}   
### solution3
{solution3}
### solution4
{solution4}
  

### Output  
'''




robustness = '''
# **Robustness Scoring Criteria (Total: 30 Points)**  
Code robustness refers to the stability and fault tolerance of code when faced with different inputs and system conditions. Good robustness means the code can handle both expected and unexpected scenarios, avoiding crashes or abnormal behavior due to exceptions, edge cases, or special inputs.  

## **1. Exception Handling (9 Points)**  
Code should effectively handle errors or exceptions to prevent crashes or unpredictable behavior.  

### **1.1 Error Capture and Handling (3 Points)**  
- **3 Points**: Code captures and handles exceptions at critical points, preventing crashes or unexpected results.  
- **2 Points**: Most exceptions are captured, but a few may be missed in some cases.  
- **1 Point**: Some exceptions are not captured, leading to system instability or crashes.  
- **0 Points**: No exception handling is implemented, causing the program to crash on errors.  

### **1.2 Exception Information Clarity (3 Points)**  
- **3 Points**: Exception messages are clear, helping developers quickly locate and understand issues.  
- **2 Points**: Exception messages are not clear or lack sufficient details, making debugging difficult.  
- **1 Point**: Exception messages are vague or missing, providing no effective debugging clues.  
- **0 Points**: No exception messages are provided, making it impossible to locate errors.  

### **1.3 Exception Rationality (3 Points)**  
- **3 Points**: Exceptions are thrown at appropriate places, and the code reacts reasonably to different error scenarios.  
- **2 Points**: Exceptions are not always thrown rationally; some cases may not require exceptions.  
- **1 Point**: Exceptions are thrown irrationally or not used, leading to system instability.  
- **0 Points**: No exception handling is implemented, causing crashes or unpredictable behavior on errors.  

## **2. Edge Cases and Special Scenario Handling (9 Points)**  
Code should correctly handle edge cases, extreme inputs, and special scenarios to ensure stability under various conditions.  

### **2.1 Edge Case Detection (3 Points)**  
- **3 Points**: Code handles all edge cases (e.g., null values, maximum/minimum values) correctly.  
- **2 Points**: Most edge cases are handled, but some extreme inputs are not fully considered.  
- **1 Point**: Some edge cases are not handled, potentially causing errors or instability.  
- **0 Points**: Edge cases are not considered, leading to crashes on extreme inputs.  

### **2.2 Special Scenario Handling (3 Points)**  
- **3 Points**: Code fully considers special scenarios (e.g., empty lists, duplicate data, invalid inputs) and handles them appropriately.  
- **2 Points**: Most special scenarios are handled, but some cases are overlooked.  
- **1 Point**: Some special scenarios are not handled effectively, potentially causing errors.  
- **0 Points**: Special scenarios are not considered, leading to crashes or abnormal behavior on special inputs.  

### **2.3 Input Validation (3 Points)**  
- **3 Points**: Code validates all inputs to ensure data legality and reasonableness.  
- **2 Points**: Some inputs are not validated, potentially allowing invalid data into the system.  
- **1 Point**: Input validation is insufficient, leading to errors from invalid inputs.  
- **0 Points**: No input validation is performed, causing crashes or unpredictable results on invalid inputs.  

## **3. Fault Tolerance (6 Points)**  
Code should handle unexpected scenarios gracefully, ensuring recovery from incorrect data or other issues.  

### **3.1 Exception Recovery (3 Points)**  
- **3 Points**: Code can recover effectively after exceptions, such as rolling back operations, retrying, or using default values.  
- **2 Points**: Some exceptions are recovered, but recovery strategies are incomplete or inconsistent.  
- **1 Point**: No effective recovery mechanism exists after exceptions, leading to instability.  
- **0 Points**: No recovery measures are taken after exceptions, causing program failure or crashes.  

### **3.2 System Fault Tolerance (3 Points)**  
- **3 Points**: The system can continue running when some components fail, ensuring critical functionality remains unaffected.  
- **2 Points**: Most failure scenarios are handled, but some faults may cause functionality to become unavailable.  
- **1 Point**: The system has low fault tolerance; failure in one module may affect the entire system.  
- **0 Points**: The system has no fault tolerance; any failure causes a crash.  

## **4. Resource Management (6 Points)**  
Code should manage resources (e.g., memory, files, network connections) properly to avoid leaks or misuse.  

### **4.1 Resource Release (3 Points)**  
- **3 Points**: All used resources are properly released (e.g., file handles, database connections).  
- **2 Points**: Most resources are released, but some cases may miss resource release.  
- **1 Point**: Resource management is poor, risking resource leaks.  
- **0 Points**: No resource release is performed, leading to resource leaks.  

### **4.2 Memory Management (3 Points)**  
- **3 Points**: Memory management is excellent, with no memory leaks or unnecessary memory usage.  
- **2 Points**: Some memory is not released, causing minor memory leaks.  
- **1 Point**: Memory management is poor, leading to memory leaks or overflow issues.  
- **0 Points**: No memory management is performed, causing memory leaks or excessive usage.  

# Task  
Your task is to rank the Robustness of the four solutions based on the Robustness Scoring Criteria above. Use the criteria to score the four solutions below and provide the final ranking. If there are solutions with the same score during sorting, please make your own judgment.  

## Input format   
[solution1]
[solution2]
[solution3]
[solution4]
 
 
       

### Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
  "solution1": {{
    "exception_handling": {{
      "total_score": ,
      "error_capture_handling": {{
        "score": ,
        "reason": ""
      }},
      "exception_message_clarity": {{
        "score": ,
        "reason": ""
      }},
      "reasonableness_of_exceptions": {{
        "score": ,
        "reason": ""
      }}
    }},
    "boundary_conditions_special_cases": {{
      "total_score": ,
      "boundary_condition_detection": {{
        "score": ,
        "reason": ""
      }},
      "special_case_handling": {{
        "score": ,
        "reason": ""
      }},
      "input_validation": {{
        "score": ,
        "reason": ""
      }}
    }},
    "fault_tolerance": {{
      "total_score": ,
      "exception_recovery": {{
        "score": ,
        "reason": ""
      }},
      "system_fault_tolerance": {{
        "score": ,
        "reason": ""
      }}
    }},
    "resource_management": {{
      "total_score": ,
      "resource_release": {{
        "score": ,
        "reason": ""
      }},
      "memory_management": {{
        "score": ,
        "reason": ""
      }}
    }},
    "solution_final_score": 
  }},
  "solution2": {{
    ...Same as above...
  }},
  "solution3": {{
    ...Same as above...
  }},
  "solution4": {{
    ...Same as above...
  }},

}}


# Annotation  
## Code problem
{code_problem} 
## Assistant’s Response  
### solution1  
{solution1}   
### solution2  
{solution2}   
### solution3
{solution3}
### solution4
{solution4}
  

### Output  
'''

functionality = '''
# Functional Suitability Scoring Criteria (Total 30 Points)

Code functionality suitability refers to whether the code accurately and completely implements the intended functionality, ensuring that all modules operate correctly according to requirements and can run stably under various inputs and edge cases, producing the expected output.

## Completeness of Function Implementation (12 Points)

This scoring item focuses on evaluating whether the code fully implements all the required functions as per the problem description and whether any key functional modules are missing.

### Coverage of Functional Modules (3 Points)

- 3 Points: The code implements all the required functions without omissions, fully meeting the requirements.
- 2 Points: The code implements most functions, but a few requirements are not covered.
- 1 Point: The code implements some functions, but multiple requirements are not met.
- 0 Points: The code only implements part of the core functions, with many requirements unmet.

### Achievement of Task Goals (3 Points)

- 3 Points: The code clearly executes according to the task goals, producing correct results as expected.
- 2 Points: The code works correctly in most cases, but some boundary or special cases are not handled.
- 1 Point: The code has obvious execution errors, and the task goal cannot be achieved.
- 0 Points: The code does not execute according to the task goals and cannot produce correct results.

### Consistency of Functional Logic (3 Points)

- 3 Points: The code maintains consistent logic and structure across all functional modules and operates as expected.
- 2 Points: Some functional modules are logically consistent, but there are a few inconsistencies.
- 1 Point: There are inconsistencies in the logic across multiple functional modules, affecting overall functionality.
- 0 Points: Lack of logical consistency between functional modules, causing the functions to fail.

### Handling of Boundary Cases (3 Points)

- 3 Points: The code correctly handles all boundary cases required by the problem, including extreme values, empty inputs, and special cases, ensuring stable program execution.
- 2 Points: The code handles most boundary cases, but some special or extreme cases are not considered.
- 1 Point: The code only handles common boundary cases, overlooking some special situations, which could lead to errors.
- 0 Points: The code fails to consider or incorrectly handles boundary cases, which may cause crashes or inaccurate results.

## Output Meets Expectations (12 Points)

This scoring item evaluates whether the generated output meets the requirements of the problem, with accurate, complete, and clear content.

### Output Accuracy (3 Points)

- 3 Points: The output fully meets the problem requirements, with no errors.
- 2 Points: The output is mostly correct, with minor discrepancies in some cases.
- 1 Point: The output shows significant deviations and does not fully meet the problem requirements.
- 0 Points: The output seriously deviates from the problem requirements and is unusable.

### Output Completeness (3 Points)

- 3 Points: The output is complete, containing all the required information without omissions.
- 2 Points: The output is mostly complete, but some key information is missing.
- 1 Point: The output lacks critical information, affecting the completeness of the problem requirements.
- 0 Points: The output is incomplete, missing a large amount of information and unable to meet the problem requirements.

### Output Clarity (3 Points)

- 3 Points: The output is clear and easy to understand, adhering to the format described in the problem.
- 2 Points: The output is generally clear, but there are some minor formatting or expression issues.
- 1 Point: The output is unclear and hard to understand, with formatting issues.
- 0 Points: The output is messy and difficult to understand, severely affecting usability.

### Output Consistency (3 Points)

- 3 Points: The output format and content are consistently aligned with the problem requirements, easy to understand, and unambiguous.
- 2 Points: The output format or content has minor inconsistencies in some cases, but it does not affect understanding or results.
- 1 Point: The output format or content has obvious inconsistencies, which may cause some degree of confusion.
- 0 Points: The output format and content are seriously inconsistent, making it difficult to understand or properly parse.

## Functional Correctness (6 Points)

This scoring item focuses on evaluating whether the code executes as expected, ensuring that each functional module operates correctly during actual execution.

### Functional Execution Correctness (3 Points)

- 3 Points: All functional modules of the code execute correctly, completing all tasks as required by the problem.
- 2 Points: Most functions execute correctly, but some modules behave abnormally in certain cases.
- 1 Point: The code has obvious errors that prevent the functions from executing correctly.
- 0 Points: The code cannot complete the core tasks and fails to implement basic functionality.

### Functional Execution Stability (3 Points)

- 3 Points: The code remains stable across multiple executions, with no crashes or exceptions.
- 2 Points: The code encounters minor errors or exceptions in some cases, but it does not affect most use cases.
- 1 Point: The code frequently encounters errors or exceptions, severely affecting stability.
- 0 Points: The code frequently crashes or causes exceptions, making it unstable.

# Task  
Your task is to rank the Functionality of the four solutions based on the Functionality Scoring Criteria above. Use the criteria to score the four solutions below and provide the final ranking. If there are solutions with the same score during sorting, please make your own judgment.  

## Input format   
[solution1]
[solution2]
[solution3]
[solution4]
 
 
       

### Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
  "solution1": {{
    "function_implementation_completeness": {{
      "total_score": ,
      "coverage_of_functional_modules": {{
        "score": ,
        "reason": ""
      }},
      "achievement_of_task_goals": {{
        "score": ,
        "reason": ""
      }},
      "consistency_of_functional_logic": {{
        "score": ,
        "reason": ""
      }},
      "handling_of_boundary_cases": {{
        "score": ,
        "reason": ""
      }}
    }},
    "output_meets_expectations": {{
      "total_score": ,
      "output_accuracy": {{
        "score": ,
        "reason": ""
      }},
      "output_completeness": {{
        "score": ,
        "reason": ""
      }},
      "output_clarity": {{
        "score": ,
        "reason": ""
      }},
      "output_consistency": {{
        "score": ,
        "reason": ""
      }}
    }},
    "functional_correctness": {{
      "total_score": ,
      "functional_execution_correctness": {{
        "score": ,
        "reason": ""
      }},
      "functional_execution_stability": {{
        "score": ,
        "reason": ""
      }}
    }},
    "solution_final_score": 
  }},
  "solution2": {{
    ...Same as above...
  }},
  "solution3": {{
    ...Same as above...
  }},
  "solution4": {{
    ...Same as above...
  }},

}}



# Annotation  
## Code problem
{code_problem} 
## Assistant’s Response  
### solution1  
{solution1}   
### solution2  
{solution2}   
### solution3
{solution3}
### solution4
{solution4}
  

### Output  
'''


standardization = '''
# Code Standardization Scoring Criteria (Total 30 Points)

Code standardization refers to writing code that adheres to established standards and best practices in terms of structure, naming, formatting, comments, and error handling to ensure consistency.

## Naming Conventions (9 Points)

Naming conventions include whether variables, functions, classes, etc., follow the project's prescribed naming rules, while also being descriptive and consistent.

### Variable Naming (3 Points)

- 3 Points: Variable names are concise, clear, follow team naming conventions, and accurately reflect their meaning.
- 2 Points: Variable names mostly follow the rules, but some may be less intuitive or slightly vague.
- 1 Point: Variable names are not standardized, and some are difficult to understand, affecting code readability.
- 0 Points: Variable names are completely inconsistent with the rules and hard to understand.

### Function/Method Naming (3 Points)

- 3 Points: Function names are concise, descriptive, clearly communicate their functionality, and follow naming conventions.
- 2 Points: Function names generally follow the rules, but some names may not be as concise or clear.
- 1 Point: Function names are unclear and do not accurately reflect their functionality, impacting understanding.
- 0 Points: Function names do not follow conventions and cannot be understood from the name.

### Class Naming (3 Points)

- 3 Points: Class names follow conventions and clearly express the function or role of the class.
- 2 Points: Class names mostly follow conventions, but some expressions may be slightly unclear.
- 1 Point: Class names do not follow conventions, and names are vague or misleading.
- 0 Points: Class names are not standardized, and the purpose of the class is unclear.

## 2. Code Structure and Formatting (9 Points)

The structure of the code should be simple and clear, following the team's coding style to enhance readability and maintainability.

### Indentation and Formatting (3 Points)

- 3 Points: The code has consistent indentation and follows the conventions, with neat formatting that is easy to read.
- 2 Points: Most indentation is correct, but some places have inconsistent indentation.
- 1 Point: The code has inconsistent indentation and messy formatting, which affects readability.
- 0 Points: The code has chaotic formatting and lacks uniform indentation, severely impacting readability.

### Code Modularization (3 Points)

- 3 Points: The code is clearly modularized, with each functional module being independent and easy to understand.
- 2 Points: The modularization is generally clear, but some parts may have redundancy or repetition.
- 1 Point: The modularization is unclear, with mixed functional code that is difficult to maintain.
- 0 Points: The code lacks modularization, and functions are mixed, making it hard to understand and maintain.

### Blank Lines and Comments (3 Points)

- 3 Points: Blank lines and comments are used appropriately, keeping the code clean and easy to understand.
- 2 Points: The use of blank lines and comments is mostly appropriate, but some areas may have too many or too few.
- 1 Point: The use of blank lines and comments is not standardized, which reduces readability.
- 0 Points: The code lacks blank lines or comments, making it difficult to read.

## 3. Error Handling Standards (6 Points)

Error handling should be reasonable and consistent, ensuring the program provides useful information and runs stably when issues arise.

### Exception Handling (3 Points)

- 3 Points: The code uses a standard exception handling mechanism, and the exception handling is reasonable and clear, providing useful debugging information.
- 2 Points: The exception handling is generally appropriate, but some exceptions may not be fully considered or are handled improperly.
- 1 Point: The exception handling is unclear and may lead to inaccurate or missing error messages.
- 0 Points: The code does not handle exceptions correctly, causing crashes or unpredictable behavior.

### Exception Information (3 Points)

- 3 Points: The exception information is clear, providing sufficient contextual information for locating the problem.
- 2 Points: The exception information is generally clear, but some key details may be missing in certain cases.
- 1 Point: The exception information is unclear or lacks effective details, making it difficult to debug.
- 0 Points: The exception information is missing or extremely vague, providing no useful debugging clues.

## 4. Commenting Standards (6 Points)

Comments should follow a unified standard, providing clear descriptions of functionality and logic, avoiding redundancy and ambiguity.

### Comment Format (3 Points)

- 3 Points: Comment format follows project standards (e.g., Javadoc, Python Docstring), with clear and concise comments.
- 2 Points: Comment format generally follows the standards, but there are minor inconsistencies in some areas.
- 1 Point: Comment format is not standardized, with some comments being unclear or chaotic.
- 0 Points: Comment format is non-standard, severely affecting code readability.

### Comment Content (3 Points)

- 3 Points: Comments accurately describe the functionality and key logic of the code, without redundancy or omissions.
- 2 Points: Comments are generally accurate, but some are too simplistic or not detailed enough.
- 1 Point: Comments are unclear or lack explanations of important code sections.
- 0 Points: Comments are missing or completely inaccurate, making it difficult to understand the code. 

# Task  
Your task is to rank the Standardization of the four solutions based on the Standardization Scoring Criteria above. Use the criteria to score the four solutions below and provide the final ranking. If there are solutions with the same score during sorting, please make your own judgment.  

## Input format  
[solution1]
[solution2]
[solution3]
[solution4]
 
 
       

### Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.
{{
  "solution1": {{
    "naming_conventions": {{
      "total_score": ,
      "variable_naming": {{
        "score": ,
        "reason": ""
      }},
      "function_method_naming": {{
        "score": ,
        "reason": ""
      }},
      "class_naming": {{
        "score": ,
        "reason": ""
      }}
    }},
    "code_structure_and_formatting": {{
      "total_score": ,
      "indentation_and_formatting": {{
        "score": ,
        "reason": ""
      }},
      "code_modularization": {{
        "score": ,
        "reason": ""
      }},
      "blank_lines_and_comments": {{
        "score": ,
        "reason": ""
      }}
    }},
    "error_handling_standards": {{
      "total_score": ,
      "exception_handling": {{
        "score": ,
        "reason": ""
      }},
      "exception_information": {{
        "score": ,
        "reason": ""
      }}
    }},
    "commenting_standards": {{
      "total_score": ,
      "comment_format": {{
        "score": ,
        "reason": ""
      }},
      "comment_content": {{
        "score": ,
        "reason": ""
      }}
    }},
    "solution_final_score": 
  }},
  "solution2": {{
    ...Same as above...
  }},
  "solution3": {{
    ...Same as above...
  }},
  "solution4": {{
    ...Same as above...
  }},

}}



# Annotation  
## Code problem
{code_problem} 
## Assistant’s Response  
### solution1  
{solution1}   
### solution2  
{solution2}   
### solution3
{solution3}
### solution4
{solution4}

  ### Output  
'''

ROBUSTNESS_AI_TEST ="""
# Task
You will receive a code problem and four solutions. Please strictly simulate the execution based on the test case and evaluate the robustness of the four solutions. Provide the assessment results, with each test case scored between 0 and 3 points. The total score cannot exceed 30 points.
## Input format   
[solution1]
[solution2]
[solution3]
[solution4]

## Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.Note the use of double quotes.
{{
  "solution1": {{
    
    "test_case1":,
    "test_case2":,
    "test_case3":,
    "test_case4":,
    "test_case5":,
    "test_case6":,
    "test_case7":,
    "test_case8":,
    "test_case9":,
    "test_case10":,
    "solution_final_score": ,
  }},
  "solution2": {{
    ...Same as above...
  }},
  "solution3": {{
    ...Same as above...
  }},
  "solution4": {{
    ...Same as above...
  }},

}}

# Annotation  
## Code problem
{code_problem} 
## Assistant’s Response  
### solution1  
{solution1}   
### solution2  
{solution2}   
### solution3
{solution3}
### solution4
{solution4}
### test cases
{ai_test}

### Output  

"""
FUNCTIONALITY_AI_TEST ="""
# Task
You will receive a code problem and four solutions. Please strictly simulate the execution based on the test case and evaluate the functionality of the four solutions. Provide the assessment results, with each test case scored between 0 and 3 points. The total score cannot exceed 30 points.
## Input format    
[solution1]
[solution2]
[solution3]
[solution4]

## Output format   
Please strictly follow the JSON format in the example below, and make sure that the brackets are properly balanced and closed.Note the use of double quotes.
{{
  "solution1": {{
    
    "test_case1":,
    "test_case2":,
    "test_case3":,
    "test_case4":,
    "test_case5":,
    "test_case6":,
    "test_case7":,
    "test_case8":,
    "test_case9":,
    "test_case10":,
    "solution_final_score": ,
  }},
  "solution2": {{
    ...Same as above...
  }},
  "solution3": {{
    ...Same as above...
  }},
  "solution4": {{
    ...Same as above...
  }},

}}

# Annotation  
## Code problem
{code_problem} 
## Assistant’s Response  
### solution1  
{solution1}   
### solution2  
{solution2}   
### solution3
{solution3}
### solution4
{solution4}
### test cases
{ai_test}

### Output
"""

class SystemPrompts:
    def __init__(self, **agents):
        self.agents = agents

    def get_agent(self, agent_name):
        return self.agents.get(agent_name, None)
    

class UserPrompts:
    def __init__(self, **prompts):
        self.prompts = prompts

    def get_prompt(self, prompt_name):
        return self.prompts.get(prompt_name, None)
USER_PROMPTS = UserPrompts(
    comment=comment,
    efficiency=efficiency,
    modularity=modularity,
    simplicity=simplicity,
    robustness=robustness,
    functionality=functionality,
    standardization=standardization,
    robustness_ai_test=ROBUSTNESS_AI_TEST,
    functionality_ai_test=FUNCTIONALITY_AI_TEST,
)
