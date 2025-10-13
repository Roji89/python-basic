from typing import TypedDict

#!/usr/bin/env python3
"""
Advanced Python Learning System - 20 Progressive Exercises
Goal: Build real skills for AI/ML development

Complete all 20 exercises, then ask me to check your solutions!
Each exercise builds toward practical AI/ML skills.
"""

# =============================================================================
# 20 PROGRESSIVE EXERCISES - Complete ALL before asking for review
# =============================================================================

"""
EXERCISE 1: Data Structure Manipulation
Create a function that takes a list of dictionaries representing students
and returns the top 3 students by grade average.

Example input: [{'name': 'Alice', 'grades': [85, 90, 78]}, {'name': 'Bob', 'grades': [92, 88, 95]}]
Expected output: List of top 3 students with their averages
"""
class Student(TypedDict):
    name: str
    grades: int
class TopStudent(TypedDict):
    name: str
    average_grade: int

def get_top_students(students_data: list[Student]) -> list[Student]:
    top_students:TopStudent = []
    for strudent in students_data:
        average_grade = sum(strudent['grades'])/ len(strudent['grades'])
        for top_student in top_students:
            if top_student == []:
                top_students.append({
                    'name': strudent['name'],
                    'average_grade': average_grade
                })
            else:
              top_student['average_grade'] < average_grade
              top_student['average_grade'] = average_grade
              top_student['name'] = strudent['name']

    pass

# =============================================================================

"""
EXERCISE 2: File Processing with Error Handling
Write a function that reads a CSV file, handles missing data, 
and returns a cleaned dataset as a list of dictionaries.
Include proper error handling for file not found, permission errors, etc.
"""

def process_csv_file(filename):
    # Your code here
    pass

# =============================================================================

"""
EXERCISE 3: Class Design for Data Analysis
Create a DataAnalyzer class with methods to:
- Load data from multiple sources (CSV, JSON, list)
- Calculate statistics (mean, median, mode, std deviation)
- Filter data based on conditions
- Export results to different formats
"""

class DataAnalyzer:
    def __init__(self):
        # Your code here
        pass
    
    def load_data(self, source, data_type='csv'):
        # Your code here
        pass
    
    def calculate_stats(self, column):
        # Your code here
        pass
    
    def filter_data(self, condition_func):
        # Your code here
        pass

# =============================================================================

"""
EXERCISE 4: Advanced List Comprehensions
Create functions using list comprehensions for:
1. Flattening nested lists
2. Creating a matrix transpose
3. Filtering and transforming data in one line
"""

def flatten_nested_list(nested_list):
    # Use list comprehension to flatten
    pass

def transpose_matrix(matrix):
    # Use list comprehension to transpose
    pass

def process_sales_data(sales_data):
    # Transform: [{'product': 'A', 'price': 100, 'qty': 5}] 
    # To: [{'product': 'A', 'revenue': 500, 'category': 'high/low'}]
    pass

# =============================================================================

"""
EXERCISE 5: Decorator Implementation
Create decorators for:
1. Timing function execution
2. Caching function results
3. Logging function calls with parameters
"""

def timer_decorator(func):
    # Your code here
    pass

def cache_decorator(func):
    # Your code here
    pass

def logger_decorator(func):
    # Your code here
    pass

# =============================================================================

"""
EXERCISE 6: API Data Processing
Write a function that simulates API responses and processes them:
- Handle different response formats (JSON, XML-like dict structure)
- Implement retry logic for failed requests
- Parse and normalize data from different API endpoints
"""

def process_api_responses(api_endpoints, max_retries=3):
    # Simulate multiple API calls and process responses
    # Your code here
    pass

# =============================================================================

"""
EXERCISE 7: Data Validation and Cleaning
Create a comprehensive data validation system:
- Validate email formats, phone numbers, dates
- Clean messy string data (remove extra spaces, standardize formats)
- Handle different data types and missing values
"""

def validate_and_clean_data(raw_data):
    # Your code here
    pass

# =============================================================================

"""
EXERCISE 8: Statistical Analysis Functions
Implement statistical functions without using external libraries:
- Correlation coefficient between two datasets
- Linear regression (slope, intercept, r-squared)
- Confidence intervals
"""

def calculate_correlation(x_data, y_data):
    # Your code here
    pass

def linear_regression(x_data, y_data):
    # Return slope, intercept, r_squared
    pass

def confidence_interval(data, confidence_level=0.95):
    # Your code here
    pass

# =============================================================================

"""
EXERCISE 9: Text Processing for NLP
Create text processing functions for natural language processing:
- Tokenization and cleaning
- N-gram generation
- Basic sentiment analysis using word lists
- Text similarity calculation
"""

def preprocess_text(text):
    # Clean, tokenize, remove stopwords
    pass

def generate_ngrams(text, n=2):
    # Generate n-grams from text
    pass

def calculate_text_similarity(text1, text2):
    # Calculate similarity score
    pass

# =============================================================================

"""
EXERCISE 10: Algorithm Implementation
Implement these algorithms from scratch:
- Binary search
- Merge sort
- K-means clustering (basic version)
"""

def binary_search(arr, target):
    # Your code here
    pass

def merge_sort(arr):
    # Your code here
    pass

def kmeans_clustering(data_points, k=3, max_iterations=100):
    # Basic k-means implementation
    pass

# =============================================================================

"""
EXERCISE 11: Database Simulation
Create a simple in-memory database with SQL-like operations:
- CREATE, INSERT, SELECT, UPDATE, DELETE
- JOIN operations between tables
- Indexing for faster queries
"""

class SimpleDatabase:
    def __init__(self):
        # Your code here
        pass
    
    def create_table(self, table_name, columns):
        # Your code here
        pass
    
    def insert(self, table_name, data):
        # Your code here
        pass
    
    def select(self, table_name, conditions=None, columns=None):
        # Your code here
        pass
    
    def join(self, table1, table2, join_column):
        # Your code here
        pass

# =============================================================================

"""
EXERCISE 12: Web Scraping Simulation
Simulate web scraping without actual requests:
- Parse HTML-like string structures
- Extract specific data patterns
- Handle different content formats
"""

def parse_html_structure(html_string):
    # Extract data from HTML-like strings
    pass

def extract_product_info(html_content):
    # Extract product names, prices, ratings
    pass

# =============================================================================

"""
EXERCISE 13: Machine Learning Data Preparation
Implement data preprocessing for ML:
- Feature scaling and normalization
- Encoding categorical variables
- Handling missing values with different strategies
- Train/validation/test split
"""

def preprocess_ml_data(dataset):
    # Your comprehensive preprocessing here
    pass

def encode_categorical_features(data, categorical_columns):
    # Your code here
    pass

def split_dataset(data, train_ratio=0.7, val_ratio=0.15):
    # Return train, validation, test sets
    pass

# =============================================================================

"""
EXERCISE 14: Model Evaluation Metrics
Implement ML evaluation metrics from scratch:
- Accuracy, Precision, Recall, F1-score
- ROC curve calculation
- Cross-validation
"""

def calculate_classification_metrics(y_true, y_pred):
    # Return accuracy, precision, recall, f1
    pass

def calculate_roc_curve(y_true, y_scores):
    # Calculate ROC curve points
    pass

def cross_validate(model_func, X, y, k_folds=5):
    # Implement k-fold cross-validation
    pass

# =============================================================================

"""
EXERCISE 15: Neural Network Basics
Implement a basic neural network from scratch:
- Forward propagation
- Backward propagation
- Gradient descent
"""

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Your code here
        pass
    
    def forward(self, X):
        # Forward propagation
        pass
    
    def backward(self, X, y, output):
        # Backward propagation
        pass
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        # Training loop
        pass

# =============================================================================

"""
EXERCISE 16: Time Series Analysis
Implement time series analysis functions:
- Moving averages
- Trend detection
- Seasonal decomposition
- Forecasting using simple methods
"""

def calculate_moving_average(data, window_size):
    # Your code here
    pass

def detect_trend(time_series_data):
    # Return 'increasing', 'decreasing', or 'stable'
    pass

def simple_forecast(historical_data, periods_ahead):
    # Simple forecasting method
    pass

# =============================================================================

"""
EXERCISE 17: Optimization Algorithms
Implement optimization algorithms:
- Gradient descent variants
- Genetic algorithm basics
- Simulated annealing
"""

def gradient_descent(cost_function, gradient_function, initial_params, learning_rate=0.01, iterations=1000):
    # Your code here
    pass

def genetic_algorithm(fitness_function, population_size=50, generations=100):
    # Basic genetic algorithm
    pass

# =============================================================================

"""
EXERCISE 18: Model Deployment Preparation
Create functions for model deployment:
- Model serialization/deserialization
- Input validation and preprocessing
- Batch prediction handling
- Performance monitoring
"""

def serialize_model(model, filename):
    # Save model to file
    pass

def load_and_predict(model_file, input_data):
    # Load model and make predictions
    pass

def batch_predict(model, batch_data, batch_size=32):
    # Handle batch predictions efficiently
    pass

# =============================================================================

"""
EXERCISE 19: Real-time Data Processing
Implement a data stream processor:
- Process data in chunks
- Maintain running statistics
- Detect anomalies in real-time
- Handle backpressure
"""

class StreamProcessor:
    def __init__(self, window_size=100):
        # Your code here
        pass
    
    def process_chunk(self, data_chunk):
        # Process incoming data chunk
        pass
    
    def detect_anomalies(self, new_data):
        # Real-time anomaly detection
        pass
    
    def get_current_stats(self):
        # Return current statistics
        pass

# =============================================================================

"""
EXERCISE 20: Complete ML Pipeline
Build a complete machine learning pipeline:
- Data ingestion from multiple sources
- Automated feature engineering
- Model selection and hyperparameter tuning
- Model evaluation and comparison
- Prediction with confidence intervals
"""

class MLPipeline:
    def __init__(self):
        # Your code here
        pass
    
    def ingest_data(self, sources):
        # Ingest from multiple data sources
        pass
    
    def engineer_features(self, raw_data):
        # Automated feature engineering
        pass
    
    def train_models(self, X, y):
        # Train multiple models and select best
        pass
    
    def evaluate_models(self, X_test, y_test):
        # Comprehensive model evaluation
        pass
    
    def predict_with_confidence(self, X):
        # Predictions with confidence intervals
        pass

# =============================================================================
# TESTING SECTION - Test each exercise individually as you complete them
# =============================================================================

def test_exercise_1():
    """Test Exercise 1: Data Structure Manipulation"""
    print("🧪 Testing Exercise 1: Data Structure Manipulation")
    
    test_students = [
        {'name': 'Alice', 'grades': [85, 90, 78]},
        {'name': 'Bob', 'grades': [92, 88, 95]},
        {'name': 'Charlie', 'grades': [76, 82, 88]},
        {'name': 'Diana', 'grades': [95, 97, 93]},
        {'name': 'Eve', 'grades': [88, 85, 92]}
    ]
    
    try:
        result = get_top_students(test_students)
        print(f"✅ Function executed successfully")
        print(f"📊 Result: {result}")
        
        # Verify result structure
        if isinstance(result, list) and len(result) <= 3:
            print("✅ Returns list with max 3 students")
        else:
            print("❌ Should return list with max 3 students")
            
        # Check if sorted by average (highest first)
        if len(result) >= 2:
            avg1 = result[0].get('average_grade', 0)
            avg2 = result[1].get('average_grade', 0)
            if avg1 >= avg2:
                print("✅ Results are sorted by average grade")
            else:
                print("❌ Results should be sorted by average grade (highest first)")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def test_exercise_2():
    """Test Exercise 2: File Processing with Error Handling"""
    print("🧪 Testing Exercise 2: File Processing with Error Handling")
    
    # Create a test CSV file
    test_csv_content = """name,age,city,salary
John,25,New York,50000
Jane,,Boston,60000
Bob,30,Chicago,
Alice,28,Seattle,70000
"""
    
    try:
        # Create test file
        with open('test_data.csv', 'w') as f:
            f.write(test_csv_content)
        
        result = process_csv_file('test_data.csv')
        print(f"✅ Function executed successfully")
        print(f"📊 Result: {result}")
        
        # Test with non-existent file
        result_error = process_csv_file('nonexistent.csv')
        print(f"📊 Error handling result: {result_error}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def test_exercise_3():
    """Test Exercise 3: Class Design for Data Analysis"""
    print("🧪 Testing Exercise 3: DataAnalyzer Class")
    
    try:
        analyzer = DataAnalyzer()
        print("✅ DataAnalyzer instance created successfully")
        
        # Test with sample data
        sample_data = [
            {'name': 'Alice', 'age': 25, 'salary': 50000},
            {'name': 'Bob', 'age': 30, 'salary': 60000},
            {'name': 'Charlie', 'age': 35, 'salary': 70000}
        ]
        
        # Test load_data
        analyzer.load_data(sample_data, 'list')
        print("✅ load_data method works")
        
        # Test calculate_stats (if implemented)
        # stats = analyzer.calculate_stats('salary')
        # print(f"📊 Stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def test_exercise_4():
    """Test Exercise 4: Advanced List Comprehensions"""
    print("🧪 Testing Exercise 4: Advanced List Comprehensions")
    
    try:
        # Test flatten_nested_list
        nested = [[1, 2], [3, 4, 5], [6]]
        flattened = flatten_nested_list(nested)
        print(f"📊 Flattened: {flattened}")
        
        # Test transpose_matrix
        matrix = [[1, 2, 3], [4, 5, 6]]
        transposed = transpose_matrix(matrix)
        print(f"📊 Transposed: {transposed}")
        
        # Test process_sales_data
        sales = [
            {'product': 'A', 'price': 100, 'qty': 5},
            {'product': 'B', 'price': 50, 'qty': 10}
        ]
        processed = process_sales_data(sales)
        print(f"📊 Processed sales: {processed}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def test_exercise_5():
    """Test Exercise 5: Decorator Implementation"""
    print("🧪 Testing Exercise 5: Decorators")
    
    try:
        # Test function to decorate
        @timer_decorator
        @cache_decorator
        @logger_decorator
        def sample_function(x, y):
            return x + y
        
        result1 = sample_function(5, 3)
        result2 = sample_function(5, 3)  # Should use cache
        print(f"📊 Function results: {result1}, {result2}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def test_exercise_6():
    """Test Exercise 6: API Data Processing"""
    print("🧪 Testing Exercise 6: API Data Processing")
    
    try:
        api_endpoints = [
            'https://api.example.com/users',
            'https://api.example.com/products',
            'https://api.example.com/orders'
        ]
        
        result = process_api_responses(api_endpoints)
        print(f"📊 API processing result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def test_exercise_7():
    """Test Exercise 7: Data Validation and Cleaning"""
    print("🧪 Testing Exercise 7: Data Validation and Cleaning")
    
    try:
        messy_data = [
            {'email': '  ALICE@EXAMPLE.COM  ', 'phone': '123-456-7890', 'date': '2023-01-15'},
            {'email': 'invalid-email', 'phone': '(555) 123-4567', 'date': '15/01/2023'},
            {'email': 'bob@test.com', 'phone': '5551234567', 'date': '2023/01/15'}
        ]
        
        cleaned = validate_and_clean_data(messy_data)
        print(f"📊 Cleaned data: {cleaned}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def test_exercise_8():
    """Test Exercise 8: Statistical Analysis Functions"""
    print("🧪 Testing Exercise 8: Statistical Analysis")
    
    try:
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 6, 8, 10]
        
        correlation = calculate_correlation(x_data, y_data)
        print(f"📊 Correlation: {correlation}")
        
        slope, intercept, r_squared = linear_regression(x_data, y_data)
        print(f"📊 Linear regression: slope={slope}, intercept={intercept}, r²={r_squared}")
        
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ci = confidence_interval(data)
        print(f"📊 Confidence interval: {ci}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def test_exercise_9():
    """Test Exercise 9: Text Processing for NLP"""
    print("🧪 Testing Exercise 9: Text Processing for NLP")
    
    try:
        text1 = "Hello, world! This is a sample text for testing."
        text2 = "Hello world! This is another sample text for testing purposes."
        
        processed = preprocess_text(text1)
        print(f"📊 Processed text: {processed}")
        
        ngrams = generate_ngrams(text1, 2)
        print(f"📊 2-grams: {ngrams}")
        
        similarity = calculate_text_similarity(text1, text2)
        print(f"📊 Text similarity: {similarity}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

def test_exercise_10():
    """Test Exercise 10: Algorithm Implementation"""
    print("🧪 Testing Exercise 10: Algorithms")
    
    try:
        # Test binary search
        arr = [1, 3, 5, 7, 9, 11, 13]
        target = 7
        index = binary_search(arr, target)
        print(f"📊 Binary search result: {index}")
        
        # Test merge sort
        unsorted = [64, 34, 25, 12, 22, 11, 90]
        sorted_arr = merge_sort(unsorted.copy())
        print(f"📊 Merge sort result: {sorted_arr}")
        
        # Test k-means
        data_points = [(1, 2), (2, 3), (8, 9), (9, 10), (10, 11)]
        clusters = kmeans_clustering(data_points, k=2)
        print(f"📊 K-means clusters: {clusters}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

# Add test functions for exercises 11-20...
def test_exercise_11():
    """Test Exercise 11: Database Simulation"""
    print("🧪 Testing Exercise 11: Database Simulation")
    
    try:
        db = SimpleDatabase()
        print("✅ Database instance created")
        
        # Test creating table
        db.create_table('users', ['id', 'name', 'email'])
        print("✅ Table created")
        
        # Test inserting data
        db.insert('users', {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'})
        print("✅ Data inserted")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 50)

# Function to run individual tests
def run_test(exercise_number):
    """Run test for a specific exercise"""
    test_functions = {
        1: test_exercise_1,
        2: test_exercise_2,
        3: test_exercise_3,
        4: test_exercise_4,
        5: test_exercise_5,
        6: test_exercise_6,
        7: test_exercise_7,
        8: test_exercise_8,
        9: test_exercise_9,
        10: test_exercise_10,
        11: test_exercise_11,
        # Add more as needed
    }
    
    if exercise_number in test_functions:
        print(f"\n🎯 Running test for Exercise {exercise_number}")
        print("=" * 60)
        test_functions[exercise_number]()
    else:
        print(f"❌ Test for Exercise {exercise_number} not implemented yet")

def run_all_tests():
    """
    Run all available tests
    """
    print("🧪 Running ALL tests...")
    print("=" * 60)
    
    for i in range(1, 12):  # Update range as you add more tests
        run_test(i)
    
    print("✅ All tests completed!")

# Quick test runners for convenience
def quick_test(exercise_num):
    """Quick way to test an exercise: quick_test(1)"""
    run_test(exercise_num)

if __name__ == "__main__":
    print("🚀 Advanced Python Learning System")
    print("📚 Complete all 20 exercises, then ask for review!")
    print("💡 Each exercise builds real AI/ML skills")
    print("\n" + "=" * 50)
    print("🧪 TESTING INSTRUCTIONS:")
    print("=" * 50)
    print("• After completing each exercise, test it immediately:")
    print("  Example: run_test(1)  # Test exercise 1")
    print("  Example: quick_test(1)  # Same as above")
    print("\n• To test all completed exercises:")
    print("  Example: run_all_tests()")
    print("\n• Available individual tests:")
    print("  run_test(1) through run_test(11) (more coming)")
    print("\n🎯 Start with Exercise 1, implement it, then run: run_test(1)")
    print("=" * 50)