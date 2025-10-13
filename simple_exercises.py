from typing import TypedDict
class Student(TypedDict):
    name: str
    grades: int
class TopStudent(TypedDict):
    name: str
    average_grade: int

#!/usr/bin/env python3
"""
Simple Python Exercises for AI/ML Preparation
Complete each exercise and test with the provided examples
"""

# =============================================================================
# EXERCISE 1: Data Structure Manipulation
# =============================================================================
"""
Create a function that takes a list of students with grades 
and returns the top 3 students by average grade.
"""

# Test data for Exercise 1
test_students = [
    {'name': 'Alice', 'grades': [85, 90, 78]},
    {'name': 'Bob', 'grades': [92, 88, 95]},
    {'name': 'Charlie', 'grades': [76, 82, 88]},
    {'name': 'Diana', 'grades': [95, 97, 93]},
    {'name': 'Eve', 'grades': [88, 85, 92]}
]

def get_top_students(students_data):
    all_student = []

    for student in students_data:
        average_grade = sum(student['grades']) / len(student['grades'])
        all_student.append({
            'name': student['name'],
            'average_grade': average_grade
        })
    top_students = sorted(all_student, key=lambda x: x['average_grade'], reverse=True)[:3]
        
    return top_students

# print("Exercise 1 Result:", get_top_students(test_students))

stocks = [
    {'symbol': 'AAPL', 'start_price': 150, 'end_price': 180},
    {'symbol': 'GOOGL', 'start_price': 2800, 'end_price': 3100},
    {'symbol': 'TSLA', 'start_price': 900, 'end_price': 1200},
    {'symbol': 'MSFT', 'start_price': 300, 'end_price': 350},
    {'symbol': 'AMZN', 'start_price': 3200, 'end_price': 3400}
]

def get_top_stocks(stocks):
    gains = []
    for stock in stocks:
        gain = (stock['end_price'] - stock['start_price']) / stock['start_price'] * 100
        gains.append({'symbol': stock['symbol'], 'gain': gain})
    top_stock = sorted(gains, key=lambda x: x['gain'], reverse=True)[:3]
    return top_stock 

# print("Stocks Result:", get_top_stocks(stocks))
# =============================================================================
# EXERCISE 2: File Processing with Error Handling
# =============================================================================
"""
Write a function that reads a CSV-like string, handles missing data,
and returns clean data as a list of dictionaries.
"""

def process_csv_data(csv_string):
    lines = csv_string.strip().split('\n')
    header = lines[0].split(',')
    result = []
    
    for line in lines[1:]:
        values = line.split(',')
        for i in range(len(values)):
            if values[i] == '':
                values[i] = None
        
        # Create one dictionary per row
        row_dict = {column: value for column, value in zip(header, values) if value is not ''}
        result.append(row_dict)
    
    return result    

# Test data for Exercise 2
test_csv = """name,age,salary
John,25,50000
Jane,,60000
Bob,30,
Alice,28,70000"""

# Test Exercise 2
print("Exercise 2 Result:", process_csv_data(test_csv))

# =============================================================================
# EXERCISE 3: List Comprehensions
# =============================================================================
"""
Create functions using list comprehensions:
1. Flatten nested lists
2. Transform sales data (add revenue calculation)
"""

def flatten_list(nested_list):
    # Your code here using list comprehension
    pass

def calculate_revenue(sales_data):
    # Add 'revenue' field to each item (price * qty)
    # Your code here using list comprehension
    pass

# Test data for Exercise 3
test_nested = [[1, 2], [3, 4, 5], [6, 7]]
test_sales = [
    {'product': 'A', 'price': 100, 'qty': 5},
    {'product': 'B', 'price': 50, 'qty': 10},
    {'product': 'C', 'price': 75, 'qty': 3}
]

# Test Exercise 3
# print("Exercise 3a Result:", flatten_list(test_nested))
# print("Exercise 3b Result:", calculate_revenue(test_sales))

# =============================================================================
# EXERCISE 4: Text Processing
# =============================================================================
"""
Create functions for text processing:
1. Clean and tokenize text (remove punctuation, lowercase, split)
2. Count word frequency
"""

def clean_and_tokenize(text):
    # Your code here
    pass

def word_frequency(text):
    # Return dictionary with word counts
    # Your code here
    pass

# Test data for Exercise 4
test_text = "Hello, World! This is a test. Hello again, world!"

# Test Exercise 4
# print("Exercise 4a Result:", clean_and_tokenize(test_text))
# print("Exercise 4b Result:", word_frequency(test_text))

# =============================================================================
# EXERCISE 5: Statistics Without Libraries
# =============================================================================
"""
Implement basic statistics functions without using external libraries:
1. Calculate mean, median, mode
2. Calculate correlation between two lists
"""

def calculate_stats(numbers):
    # Return dict with mean, median, mode
    # Your code here
    pass

def correlation(x_data, y_data):
    # Calculate correlation coefficient
    # Your code here
    pass

# Test data for Exercise 5
test_numbers = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
test_x = [1, 2, 3, 4, 5]
test_y = [2, 4, 6, 8, 10]

# Test Exercise 5
# print("Exercise 5a Result:", calculate_stats(test_numbers))
# print("Exercise 5b Result:", correlation(test_x, test_y))

# =============================================================================
# EXERCISE 6: Algorithm Implementation
# =============================================================================
"""
Implement these algorithms:
1. Binary search
2. Simple sorting algorithm
"""

def binary_search(arr, target):
    # Return index of target or -1 if not found
    # Your code here
    pass

def simple_sort(arr):
    # Sort array using any algorithm you know
    # Your code here
    pass

# Test data for Exercise 6
test_sorted_array = [1, 3, 5, 7, 9, 11, 13]
test_unsorted_array = [64, 34, 25, 12, 22, 11, 90]

# Test Exercise 6
# print("Exercise 6a Result:", binary_search(test_sorted_array, 7))
# print("Exercise 6b Result:", simple_sort(test_unsorted_array))

# =============================================================================
# EXERCISE 7: Data Validation
# =============================================================================
"""
Create a function that validates and cleans user data:
- Check email format
- Clean phone numbers
- Validate age ranges
"""

def validate_user_data(user_data):
    # Return cleaned data with validation status
    # Your code here
    pass

# Test data for Exercise 7
test_users = [
    {'email': '  ALICE@EXAMPLE.COM  ', 'phone': '123-456-7890', 'age': 25},
    {'email': 'invalid-email', 'phone': '(555) 123-4567', 'age': -5},
    {'email': 'bob@test.com', 'phone': '5551234567', 'age': 150}
]

# Test Exercise 7
# print("Exercise 7 Result:", validate_user_data(test_users))

# =============================================================================
# EXERCISE 8: Simple Machine Learning Data Prep
# =============================================================================
"""
Prepare data for machine learning:
1. Normalize numerical features (0-1 scaling)
2. Encode categorical features to numbers
"""

def normalize_features(data, feature_columns):
    # Scale features to 0-1 range
    # Your code here
    pass

def encode_categories(data, category_column):
    # Convert categories to numbers
    # Your code here
    pass

# Test data for Exercise 8
test_ml_data = [
    {'name': 'Alice', 'age': 25, 'salary': 50000, 'department': 'IT'},
    {'name': 'Bob', 'age': 30, 'salary': 60000, 'department': 'HR'},
    {'name': 'Charlie', 'age': 35, 'salary': 70000, 'department': 'IT'},
    {'name': 'Diana', 'age': 28, 'salary': 55000, 'department': 'Finance'}
]

# Test Exercise 8
# print("Exercise 8a Result:", normalize_features(test_ml_data, ['age', 'salary']))
# print("Exercise 8b Result:", encode_categories(test_ml_data, 'department'))

# =============================================================================
# EXERCISE 9: Simple Classification
# =============================================================================
"""
Implement a simple classification algorithm:
Find the most common class in training data and use it to predict
"""

def simple_classifier(training_data, test_data):
    # Find most common class in training_data
    # Apply to test_data
    # Your code here
    pass

# Test data for Exercise 9
training_data = [
    {'features': [1, 2], 'class': 'A'},
    {'features': [2, 3], 'class': 'A'},
    {'features': [3, 4], 'class': 'B'},
    {'features': [4, 5], 'class': 'A'},
    {'features': [5, 6], 'class': 'B'}
]

test_data = [
    {'features': [1.5, 2.5]},
    {'features': [3.5, 4.5]}
]

# Test Exercise 9
# print("Exercise 9 Result:", simple_classifier(training_data, test_data))

# =============================================================================
# EXERCISE 10: Data Pipeline
# =============================================================================
"""
Create a simple data processing pipeline:
Load -> Clean -> Transform -> Export
"""

def data_pipeline(raw_data):
    # Process data through multiple steps
    # Your code here
    pass

# Test data for Exercise 10
raw_data = [
    "Alice,25,engineer,50000",
    "Bob,,manager,60000",
    "Charlie,30,,55000",
    "Diana,28,analyst,52000"
]

# Test Exercise 10
# print("Exercise 10 Result:", data_pipeline(raw_data))

# =============================================================================
# HOW TO TEST YOUR SOLUTIONS
# =============================================================================
"""
After implementing each function:

1. Uncomment the test line for that exercise
2. Run the file: python3 simple_exercises.py
3. Check if your output makes sense
4. Move to the next exercise

Example:
- Implement get_top_students()
- Uncomment: print("Exercise 1 Result:", get_top_students(test_students))
- Run: python3 simple_exercises.py
- Check the output
"""

if __name__ == "__main__":
    print("🚀 Simple Python Exercises for AI/ML")
    print("✅ Implement each function, then uncomment its test line")
    print("🧪 Run 'python3 simple_exercises.py' to test")
    print("\nStart with Exercise 1: get_top_students()")
