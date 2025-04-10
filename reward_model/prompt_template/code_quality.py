COMMENT_PROMPT = """
# Code Commenting Evaluation Criteria (Total Score: 0-5)

Comments should accurately convey the purpose and logic of the code, ensuring the information is concise, easy to understand, and unambiguous. They should follow a consistent style and standard, aligning with the overall code structure while explaining key parts. Over-commenting or omission should be avoided to ensure appropriate and effective information delivery.

### Dimension1: Language Clarity 

- excellent: Comments are concise and clear, with no redundant or ambiguous expressions, and the language is fluent and smooth.
- good: Some comments contain minor redundancy or slight ambiguity but do not affect overall comprehension.
- normal: Many comments are ambiguous or overly verbose, affecting readability.
- bad: Most comments are difficult to understand and require extra effort to interpret.

### Dimension2: Terminology Usage 

- excellent: Terms are accurate, all technical terms are appropriately explained, and align with the code logic.
- good: Terminology is mostly accurate, but some lack necessary explanations.
- normal: Incorrect or inconsistent terminology may cause misunderstandings.
- bad: Terms are used arbitrarily, making comprehension difficult or misleading developers.

### Dimension3: Complex Logic Background Information 

- excellent: All complex algorithms or business logic are well-explained with background information.
- good: Most complex logic has background information, but some areas need further clarification.
- normal: Only some key logic has background explanations, requiring additional effort to understand.
- bad: Completely lacks background information, making complex logic difficult to understand.

### Dimension4: Function Description 

- excellent: Comments clearly and completely explain the function and purpose of the code block, making it understandable without reading the code.
- good: The function description is relatively clear but needs some improvements.
- normal: Only some code blocks have function descriptions, lacking overall consistency.
- bad: No function descriptions are provided.

### Dimension5: Key Logic and Algorithm Explanation 

- excellent: All complex algorithms and key logic are well-commented, including algorithm concepts and key steps, facilitating understanding and maintenance.
- good: Basic explanations are provided for algorithms or logic, but some critical points lack details.
- normal: Only input and output are described, without explaining the algorithm process or implementation idea.
- bad: No explanations for algorithms at all.

### Dimension6: Edge Cases and Exception Handling 

- excellent: Clearly describes edge cases, exception handling logic, and special condition handling.
- good: Some edge cases or exceptions are commented on, but gaps remain.
- normal: Comments exist only for specific situations, ignoring some potential issues.
- bad: No comments on edge cases or exceptions.

### Dimension7: Formatting Standards 

- excellent: Strictly follows project or industry standards (e.g., Javadoc, Python Docstring, Doxygen), maintaining a uniform format and best practices.
- good: Mostly follows formatting standards, but some inconsistencies exist.
- normal: Formatting is inconsistent, mixing multiple styles, which affects readability.
- bad: Formatting is chaotic, non-standard, and severely impacts readability.

### Dimension8: Language Consistency 

- excellent: Comments are entirely in English and maintain a consistent language style.
- good: Very few instances of mixed-language comments, but they do not impact overall understanding.
- normal: Mixed Chinese and English comments, affecting readability.
- bad: Random language switching in comments, severely impacting comprehension.

### Dimension9: Comment Density 

- excellent: The density of comments is appropriate, matching the complexity of the code logic with sufficient information, neither excessive nor lacking.
- good: Some code sections have too many or too few comments, but overall comprehension is still possible.
- normal: A large number of unnecessary comments or too few comments affect readability.
- bad: Comments are extremely sparse or entirely absent.

### Dimension10: Distracting Comments 

- excellent: No redundant, outdated, or repetitive comments; all comments are meaningful and effective.
- good: A few redundant or unnecessary comments exist, but they have minimal impact.
- normal: A noticeable amount of outdated or repetitive comments, affecting readability.
- bad: A large number of outdated or meaningless comments, severely interfering with code readability.



## Code problem
{code_problem} 
## solution
{solution} 

# Task
Your task is to evaluate the quality of code comments in the provided solution using the Code comments Scoring Criteria outlined above. 
Please provide a score from 0 to 5 , with 0 being the worst and 5 being the best.

## Output
Please provide the score of solution(0-5): <|reward|>
END

"""




EFFICIENCY_PROMPT = '''
# Efficiency Scoring Criteria 

Code efficiency refers to the ability of the code to accomplish tasks with the least resource consumption (such as time, memory, computational power, etc.) during execution. Efficient code is capable of processing large amounts of data in a short time while ensuring stable operation with limited resources, avoiding unnecessary performance bottlenecks or wastage.

### Algorithm Efficiency 

- excellent: Uses optimal or near-optimal algorithms with time complexity optimized as much as possible (e.g., O(n) instead of O(n²)).
- good: The algorithm is generally reasonable but still has room for optimization (e.g., could be replaced with a more efficient algorithm).
- normal: Uses suboptimal algorithms, causing performance degradation, but the impact is not critical.
- bad: The algorithm is inefficient, with obvious optimization needs.

### Algorithm Adaptability 

- excellent: The algorithm is highly suited to the actual application scenario, capable of efficiently handling large-scale data or high concurrency.
- good: The algorithm can generally adapt to the scenario, but efficiency is lower in some cases.
- normal: The algorithm choice is not suitable for the current application scenario, limiting performance.
- bad: The algorithm is completely unsuitable for the current needs, significantly impacting program performance.

### Redundant Computation 

- excellent: No redundant computation, the code logic is streamlined and efficient.
- good: There is minor redundant computation, but the overall impact is minimal.
- normal: Obvious redundant computation, affecting performance.
- bad: There is significant redundant computation, severely slowing down execution speed.

### Loop Optimization 

- excellent: Loop optimization is appropriate, avoiding unnecessary nesting and repeated calculations.
- good: The loops are generally reasonable, but there is still room for optimization in some parts.
- normal: Inefficient loop writing, affecting execution speed.
- bad: Large amounts of inefficient loops, severely degrading performance.

### Data Structure Choice 

- excellent: Uses the optimal data structure, reasonably matching the usage scenario.
- good: The data structure is generally suitable, but there are areas that could be optimized.
- normal: Uses suboptimal data structures, affecting storage efficiency.
- bad: The choice of data structure is clearly inappropriate, impacting performance.

### Variable and Object Management 

- excellent: Proper use of variables and objects, avoiding unnecessary memory occupation.
- good: There are a small number of redundant variables or objects, but the impact is minimal.
- normal: Multiple unnecessary variables or objects, causing resource waste.
- bad: A large number of redundant variables or objects, occupying unnecessary memory resources.

### Caching and Reuse 

- excellent: Fully utilizes caching and object reuse mechanisms, reducing unnecessary repeated creation.
- good: Some areas could optimize caching and reuse strategies, but the overall impact is minimal.
- normal: Limited use of caching, leading to repeated calculations or unnecessary object creation.
- bad: No caching used, resulting in a lot of repeated calculations or unnecessary object creation.

### Parallel and Asynchronous Optimization 

- excellent: Appropriately uses multithreading, parallel computing, or asynchronous programming to improve execution efficiency.
- good: Some scenarios could optimize parallel or asynchronous strategies, but the impact is minimal.
- normal: Parallel or asynchronous processing has obvious optimization space, affecting code efficiency.
- bad: Does not use parallel or asynchronous optimization, leading to low performance. Contains inefficient synchronous operations, severely slowing down the program's runtime.

### I/O and Database Optimization 

- excellent: Reduces I/O operations and optimizes database queries to improve access efficiency.
- good: I/O or database access still has room for optimization, but overall it is acceptable.
- normal: I/O or database query design is unreasonable, affecting program performance.
- bad: Contains large amounts of inefficient I/O or database queries, leading to significant performance issues.

### Code Redundancy 

- excellent: The code is concise, with no redundant logic, and all parts serve a practical purpose.
- good: Very few redundant parts, but the overall impact is minimal.
- normal: Some unused code, affecting code maintainability and execution efficiency.
- bad: A large amount of unused code, severely affecting code maintainability and execution efficiency.


## Code problem
{code_problem} 
## solution
{solution} 

# Task
Your task is to evaluate the code Efficiency in the provided solution using the Code Efficiency Scoring Criteria outlined above. 
Please provide a score from 0 to 5 , with 0 being the worst and 5 being the best.

## Output
Please provide the score of solution(0-5): <|reward|>
END
'''



MODULARITY_PROMPT = '''
# Modularization Scoring Criteria 

Code modularity refers to the practice of dividing code into multiple independent, reusable modules, each responsible for a specific function or task. Each module should have a clear interface and boundaries, allowing for independent development, testing, and maintenance. It also reduces the coupling between modules, enhancing the system's scalability, maintainability, and reusability.

### Single Responsibility Principle 

- excellent: Each module is responsible for a single task, following the single responsibility principle.
- good: Most modules have clear responsibilities, but some modules have multiple responsibilities.
- normal: Multiple modules take on too many different responsibilities, leading to high coupling and low readability.
- bad: Module responsibilities are confused, with a single module taking on multiple unrelated tasks, making the code hard to maintain and expand.

### Module Independence 

- excellent: Modules are highly independent, communicating only through clear interfaces.
- good: Modules are generally independent, but there are some unnecessary dependencies.
- normal: Several modules are highly coupled, lacking clear boundaries.
- bad: Modules are heavily interdependent, making it difficult to test or replace them independently, resulting in high system coupling.

### Code Organization Structure 

- excellent: File and directory structure is clear, following best practices, and module division is reasonable.
- good: Overall structure is good, but some modules are poorly positioned.
- normal: The code organization is chaotic, making it difficult to quickly understand module relationships.
- bad: The code structure is messy, with illogical module division and a lack of reasonable file and directory organization, making understanding and maintenance extremely difficult.

### Module Dependency Relationships 

- excellent: Dependencies between modules are clear, with a simple dependency graph, avoiding complex dependency chains.
- good: Module dependencies are generally clear, but there are some instances of confusing dependencies or circular dependencies.
- normal: Dependencies between modules are unclear, with some modules having complex dependency chains.
- bad: Module dependencies are complex, with numerous circular dependencies or implicit coupling, making code modifications prone to errors.

### Code Reusability Level 

- excellent: High reusability, core functionality is well encapsulated, and code reuse rate is high.
- good: Most code is reusable, but some duplicate logic still exists.
- normal: Only part of the code is reusable, with a significant amount of redundant code.
- bad: No code reuse, all functionality is implemented repeatedly, leading to a large amount of redundant code.

### Common Module Encapsulation 

- excellent: Common functionalities have been encapsulated into independent modules and applied appropriately.
- good: Some functionalities are well modularized, but there is still code that could be extracted.
- normal: Only a few functions are encapsulated, with a large amount of code not modularized.
- bad: No common module encapsulation, all functionality is written directly into business code, resulting in very poor maintainability.

### Redundant Code Elimination 

- excellent: Redundant code has been avoided, and similar functionalities have been encapsulated into generic modules.
- good: Most redundant code has been eliminated, but some similar functionalities have not been extracted.
- normal: Redundant code is severe, with a lack of reasonable modularization and extraction.
- bad: The code is filled with a large amount of redundant logic, and similar functions are not encapsulated, making the code bloated and difficult to modify.

### Interface Clarity 

- excellent: Interface names are standardized, parameters are clear, documentation is complete, and the interface is easy to understand and use.
- good: The interface is generally clear, but some parameters or names are not intuitive.
- normal: The interface design is not standardized, naming is chaotic, and usage is inconvenient.
- bad: The interface design is chaotic, naming is not standardized, there is no documentation, and the calling method is obscure and hard to understand, making the module difficult to use.

### High Cohesion 

- excellent: The internal functions of the module are highly related, with a single responsibility, tight logic, and ease of maintenance and expansion.
- good: The internal functions of the module are generally related, but there are some scattered responsibilities.
- normal: The internal functions of the module are loosely related, with unclear responsibilities, making maintenance difficult.
- bad: The internal logic of the module is chaotic, with unrelated functions, a lack of unity, making code maintenance extremely difficult.

### Low Coupling 

- excellent: Dependencies between modules are clear, interacting through interfaces, and modifying one module has minimal impact on other modules.
- good: There are minor dependencies between modules, but the overall coupling is within an acceptable range.
- normal: Modules are highly dependent on other modules, making it difficult to use or modify them independently.
- bad: Modules are severely coupled, directly accessing the internal implementation of other modules, and any modification could affect the entire system, resulting in high maintenance costs.

## Code problem
{code_problem} 
## solution
{solution} 

# Task
Your task is to evaluate the code modularity in the provided solution using the Code modularity Scoring Criteria outlined above. 
Please provide a score from 0 to 5 , with 0 being the worst and 5 being the best.

## Output
Please provide the score of solution(0-5): <|reward|>
END
'''


SIMPLICITY_PROMPT = '''
# Simplicity Scoring Criteria 

Code simplicity refers to implementing requirements with minimal code, efficient logical structures, and clear expressions while ensuring completeness and readability. It avoids redundancy, repetition, and unnecessary complexity, thereby enhancing maintainability and understandability.

### Code Depth 

- excellent: The code structure is reasonable, with appropriate depth, flat structure, and easy to track.
- good: The code has multiple levels, but overall readability is acceptable with no major difficulty in understanding.
- normal: The code has excessive depth, high complexity, and is difficult to understand and maintain.
- bad: The code depth is excessively complex with no logical structure, making it extremely difficult to understand and maintain.

### Function/Method Length 

- excellent: Functions/methods are short, with a single responsibility, and easy to understand.
- good: Function/method length is moderate, covering multiple steps but does not affect readability.
- normal: Functions/methods are too long, with multiple responsibilities, making them difficult to understand and debug.
- bad: Functions/methods are unusually long, difficult to trace, with unclear responsibilities, and difficult to understand.

### Code Duplication 

- excellent: No code duplication, all functionalities are modularized and reused, avoiding redundancy.
- good: Most code is not duplicated, but there are some repeated logic.
- normal: Code duplication is severe, requiring extensive extraction and refactoring.
- bad: Excessive code duplication, making it nearly impossible to reuse and leading to redundancy and maintenance difficulty.

### Ineffective/Redundant Code 

- excellent: No ineffective or redundant code, every part contributes to the implementation.
- good: Some redundant code exists, but it does not affect the code execution.
- normal: There is a large amount of redundant or ineffective code, which affects code simplicity and performance.
- bad: The code contains large amounts of ineffective and redundant parts, severely impacting functionality and code efficiency.

### Variable and Function Naming 

- excellent: Naming is concise and descriptive, accurately expressing the purpose, avoiding overly long or short names.
- good: Naming is generally clear, but some names are a bit unclear or too long.
- normal: Naming is not intuitive, too short or too long, hindering understanding.
- bad: Naming completely fails to meet standards, making it impossible to understand its purpose or meaning.

### Code Comments 

- excellent: Comments are concise and effective, clearly describing the purpose and key points of the code, avoiding redundant comments.
- good: Comments are relatively clear, but some parts have excessive or inaccurate comments.
- normal: Comments are unclear, lacking necessary explanations or being too verbose.
- bad: No comments or comments are completely unclear, making it difficult to understand the purpose and implementation of the code.

### Control Structure Simplicity 

- excellent: Control structures are simple and clear, avoiding unnecessary nesting and complex conditions.
- good: Control structures are generally clear, but some conditional statements or nesting are excessive.
- normal: Control structures are complex, with too much nesting, affecting readability.
- bad: Control structures are overly complex, with chaotic logic, making them hard to understand or maintain.

### Code Style Consistency 

- excellent: Code style is uniform, following consistent conventions (e.g., indentation, spaces, brackets) with no style conflicts.
- good: Code style is mostly consistent, but there are slight inconsistencies in some places.
- normal: Code style is inconsistent, with noticeable deviations or incorrect conventions.
- bad: Code style is severely inconsistent, with numerous convention errors, affecting code readability.

### Use of Advanced Language Features 

- excellent: Advanced language features (e.g., list comprehensions, lambda functions, generators) are used to simplify code and improve readability and simplicity.
- good: Some advanced language features are applied, but there is still room for simplification in some areas.
- normal: Little or no use of advanced language features, and the code can be further simplified.
- bad: No use of advanced language features, resulting in overly complex and hard-to-understand code.

### Design Patterns and Best Practices 

- excellent: Appropriate design patterns (e.g., Singleton, Factory) and best practices are used effectively to reduce redundant code.
- good: Some design patterns or best practices are applied, but not all situations are optimized.
- normal: Design patterns and best practices are underused, and the code can be further optimized and simplified.
- bad: No design patterns or best practices are used, resulting in chaotic and hard-to-expand code.

## Code problem
{code_problem} 
## solution
{solution} 

# Task
Your task is to evaluate the code Simplicity in the provided solution using the Code Simplicity Scoring Criteria outlined above. 
Please provide a score from 0 to 5 , with 0 being the worst and 5 being the best.

## Output
Please provide the score of solution(0-5): <|reward|>
END
'''


ROBUSTNESS_PROMPT = '''
# **Robustness Scoring Criteria**  
Code robustness refers to the stability and fault tolerance of code when faced with different inputs and system conditions. Good robustness means the code can handle both expected and unexpected scenarios, avoiding crashes or abnormal behavior due to exceptions, edge cases, or special inputs.  

### Error Capture and Handling
- **excellent**: Code captures and handles exceptions at critical points, preventing crashes or unexpected results.  
- **good**: Most exceptions are captured, but a few may be missed in some cases.  
- **normal**: Some exceptions are not captured, leading to system instability or crashes.  
- **bad**: No exception handling is implemented, causing the program to crash on errors.  

### Exception Information Clarity
- **excellent**: Exception messages are clear, helping developers quickly locate and understand issues.  
- **good**: Exception messages are not clear or lack sufficient details, making debugging difficult.  
- **normal**: Exception messages are vague or missing, providing no effective debugging clues.  
- **bad**: No exception messages are provided, making it impossible to locate errors.  

### Exception Rationality
- **excellent**: Exceptions are thrown at appropriate places, and the code reacts reasonably to different error scenarios.  
- **good**: Exceptions are not always thrown rationally; some cases may not require exceptions.  
- **normal**: Exceptions are thrown irrationally or not used, leading to system instability.  
- **bad**: No exception handling is implemented, causing crashes or unpredictable behavior on errors.  

### Edge Case Detection
- **excellent**: Code handles all edge cases (e.g., null values, maximum/minimum values) correctly.  
- **good**: Most edge cases are handled, but some extreme inputs are not fully considered.  
- **normal**: Some edge cases are not handled, potentially causing errors or instability.  
- **bad**: Edge cases are not considered, leading to crashes on extreme inputs.  

### Special Scenario Handling
- **excellent**: Code fully considers special scenarios (e.g., empty lists, duplicate data, invalid inputs) and handles them appropriately.  
- **good**: Most special scenarios are handled, but some cases are overlooked.  
- **normal**: Some special scenarios are not handled effectively, potentially causing errors.  
- **bad**: Special scenarios are not considered, leading to crashes or abnormal behavior on special inputs.  

### Input Validation
- **excellent**: Code validates all inputs to ensure data legality and reasonableness.  
- **good**: Some inputs are not validated, potentially allowing invalid data into the system.  
- **normal**: Input validation is insufficient, leading to errors from invalid inputs.  
- **bad**: No input validation is performed, causing crashes or unpredictable results on invalid inputs.  

### Exception Recovery
- **excellent**: Code can recover effectively after exceptions, such as rolling back operations, retrying, or using default values.  
- **good**: Some exceptions are recovered, but recovery strategies are incomplete or inconsistent.  
- **normal**: No effective recovery mechanism exists after exceptions, leading to instability.  
- **bad**: No recovery measures are taken after exceptions, causing program failure or crashes.  

### System Fault Tolerance
- **excellent**: The system can continue running when some components fail, ensuring critical functionality remains unaffected.  
- **good**: Most failure scenarios are handled, but some faults may cause functionality to become unavailable.  
- **normal**: The system has low fault tolerance; failure in one module may affect the entire system.  
- **bad**: The system has no fault tolerance; any failure causes a crash.  

### Resource Release
- **excellent**: All used resources are properly released (e.g., file handles, database connections).  
- **good**: Most resources are released, but some cases may miss resource release.  
- **normal**: Resource management is poor, risking resource leaks.  
- **bad**: No resource release is performed, leading to resource leaks.  

### Memory Management
- **excellent**: Memory management is excellent, with no memory leaks or unnecessary memory usage.  
- **good**: Some memory is not released, causing minor memory leaks.  
- **normal**: Memory management is poor, leading to memory leaks or overflow issues.  
- **bad**: No memory management is performed, causing memory leaks or excessive usage.  

## Code problem
{code_problem} 
## solution
{solution} 

# Task
Your task is to evaluate the code Robustness in the provided solution using the Code Robustness Scoring Criteria outlined above. 
Please provide a score from 0 to 5 , with 0 being the worst and 5 being the best.

## Output
Please provide the score of solution(0-5): <|reward|>
END
'''


FUNCTIONALITY_PROMPT = '''
# Functional Suitability Scoring Criteria 

Code functionality suitability refers to whether the code accurately and completely implements the intended functionality, ensuring that all modules operate correctly according to requirements and can run stably under various inputs and edge cases, producing the expected output.

### Coverage of Functional Modules 

- excellent: The code implements all the required functions without omissions, fully meeting the requirements.
- good: The code implements most functions, but a few requirements are not covered.
- normal: The code implements some functions, but multiple requirements are not met.
- bad: The code only implements part of the core functions, with many requirements unmet.

### Achievement of Task Goals 

- excellent: The code clearly executes according to the task goals, producing correct results as expected.
- good: The code works correctly in most cases, but some boundary or special cases are not handled.
- normal: The code has obvious execution errors, and the task goal cannot be achieved.
- bad: The code does not execute according to the task goals and cannot produce correct results.

### Consistency of Functional Logic 

- excellent: The code maintains consistent logic and structure across all functional modules and operates as expected.
- good: Some functional modules are logically consistent, but there are a few inconsistencies.
- normal: There are inconsistencies in the logic across multiple functional modules, affecting overall functionality.
- bad: Lack of logical consistency between functional modules, causing the functions to fail.

### Handling of Boundary Cases 

- excellent: The code correctly handles all boundary cases required by the problem, including extreme values, empty inputs, and special cases, ensuring stable program execution.
- good: The code handles most boundary cases, but some special or extreme cases are not considered.
- normal: The code only handles common boundary cases, overlooking some special situations, which could lead to errors.
- bad: The code fails to consider or incorrectly handles boundary cases, which may cause crashes or inaccurate results.

### Output Accuracy 

- excellent: The output fully meets the problem requirements, with no errors.
- good: The output is mostly correct, with minor discrepancies in some cases.
- normal: The output shows significant deviations and does not fully meet the problem requirements.
- bad: The output seriously deviates from the problem requirements and is unusable.

### Output Completeness 

- excellent: The output is complete, containing all the required information without omissions.
- good: The output is mostly complete, but some key information is missing.
- normal: The output lacks critical information, affecting the completeness of the problem requirements.
- bad: The output is incomplete, missing a large amount of information and unable to meet the problem requirements.

### Output Clarity 

- excellent: The output is clear and easy to understand, adhering to the format described in the problem.
- good: The output is generally clear, but there are some minor formatting or expression issues.
- normal: The output is unclear and hard to understand, with formatting issues.
- bad: The output is messy and difficult to understand, severely affecting usability.

### Output Consistency 

- excellent: The output format and content are consistently aligned with the problem requirements, easy to understand, and unambiguous.
- good: The output format or content has minor inconsistencies in some cases, but it does not affect understanding or results.
- normal: The output format or content has obvious inconsistencies, which may cause some degree of confusion.
- bad: The output format and content are seriously inconsistent, making it difficult to understand or properly parse.

### Functional Execution Correctness 

- excellent: All functional modules of the code execute correctly, completing all tasks as required by the problem.
- good: Most functions execute correctly, but some modules behave abnormally in certain cases.
- normal: The code has obvious errors that prevent the functions from executing correctly.
- bad: The code cannot complete the core tasks and fails to implement basic functionality.

### Functional Execution Stability 

- excellent: The code remains stable across multiple executions, with no crashes or exceptions.
- good: The code encounters minor errors or exceptions in some cases, but it does not affect most use cases.
- normal: The code frequently encounters errors or exceptions, severely affecting stability.
- bad: The code frequently crashes or causes exceptions, making it unstable.

## Code problem
{code_problem} 
## solution
{solution} 

# Task
Your task is to evaluate the code Functional Suitability in the provided solution using the Code Functional Suitability Scoring Criteria outlined above. 
Please provide a score from 0 to 5 , with 0 being the worst and 5 being the best.

## Output
Please provide the score of solution(0-5): <|reward|>
END
'''


STANDARDIZATION_PROMPT = '''
# Code Standardization Scoring Criteria 

Code standardization refers to writing code that adheres to established standards and best practices in terms of structure, naming, formatting, comments, and error handling to ensure consistency.

### Variable Naming 

- excellent: Variable names are concise, clear, follow team naming conventions, and accurately reflect their meaning.
- good: Variable names mostly follow the rules, but some may be less intuitive or slightly vague.
- normal: Variable names are not standardized, and some are difficult to understand, affecting code readability.
- bad: Variable names are completely inconsistent with the rules and hard to understand.

### Function/Method Naming 

- excellent: Function names are concise, descriptive, clearly communicate their functionality, and follow naming conventions.
- good: Function names generally follow the rules, but some names may not be as concise or clear.
- normal: Function names are unclear and do not accurately reflect their functionality, impacting understanding.
- bad: Function names do not follow conventions and cannot be understood from the name.

### Class Naming 

- excellent: Class names follow conventions and clearly express the function or role of the class.
- good: Class names mostly follow conventions, but some expressions may be slightly unclear.
- normal: Class names do not follow conventions, and names are vague or misleading.
- bad: Class names are not standardized, and the purpose of the class is unclear.

### Indentation and Formatting 

- excellent: The code has consistent indentation and follows the conventions, with neat formatting that is easy to read.
- good: Most indentation is correct, but some places have inconsistent indentation.
- normal: The code has inconsistent indentation and messy formatting, which affects readability.
- bad: The code has chaotic formatting and lacks uniform indentation, severely impacting readability.

### Code Modularization 

- excellent: The code is clearly modularized, with each functional module being independent and easy to understand.
- good: The modularization is generally clear, but some parts may have redundancy or repetition.
- normal: The modularization is unclear, with mixed functional code that is difficult to maintain.
- bad: The code lacks modularization, and functions are mixed, making it hard to understand and maintain.

### Blank Lines and Comments 

- excellent: Blank lines and comments are used appropriately, keeping the code clean and easy to understand.
- good: The use of blank lines and comments is mostly appropriate, but some areas may have too many or too few.
- normal: The use of blank lines and comments is not standardized, which reduces readability.
- bad: The code lacks blank lines or comments, making it difficult to read.

### Exception Handling 

- excellent: The code uses a standard exception handling mechanism, and the exception handling is reasonable and clear, providing useful debugging information.
- good: The exception handling is generally appropriate, but some exceptions may not be fully considered or are handled improperly.
- normal: The exception handling is unclear and may lead to inaccurate or missing error messages.
- bad: The code does not handle exceptions correctly, causing crashes or unpredictable behavior.

### Exception Information 

- excellent: The exception information is clear, providing sufficient contextual information for locating the problem.
- good: The exception information is generally clear, but some key details may be missing in certain cases.
- normal: The exception information is unclear or lacks effective details, making it difficult to debug.
- bad: The exception information is missing or extremely vague, providing no useful debugging clues.

### Comment Format 

- excellent: Comment format follows project standards (e.g., Javadoc, Python Docstring), with clear and concise comments.
- good: Comment format generally follows the standards, but there are minor inconsistencies in some areas.
- normal: Comment format is not standardized, with some comments being unclear or chaotic.
- bad: Comment format is non-standard, severely affecting code readability.

### Comment Content 

- excellent: Comments accurately describe the functionality and key logic of the code, without redundancy or omissions.
- good: Comments are generally accurate, but some are too simplistic or not detailed enough.
- normal: Comments are unclear or lack explanations of important code sections.
- bad: Comments are missing or completely inaccurate, making it difficult to understand the code.

## Code problem
{code_problem} 
## solution
{solution} 

# Task
Your task is to evaluate the code Standardization in the provided solution using the Code Standardization Scoring Criteria outlined above. 
Please provide a score from 0 to 5 , with 0 being the worst and 5 being the best.

## Output
Please provide the score of solution(0-5): <|reward|>
END
'''