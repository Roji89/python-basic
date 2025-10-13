print('Hello to shiny day dada')
# number1 =  input('Enter first number: ')
# number2 =  input('Enter second number: ')
# sum = int(number1) + int(number2)
# print('The sum of the two numbers is: ', sum)

friends = ['Alice', 'Bob', 'Charlie1']
for friend in friends:
    print('Hello, ' + friend + '!')

friends.append('David')
print('Updated friends list:', friends)
friends.remove('Alice')
print('After removing Alice:', friends)
friends.insert(0, 'Roja')
print('After inserting Roja at the beginning:', friends)
friends.sort(reverse=True)
print('After sorting alphabetically:', friends)
friends.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf'))
print('After sorting by number if present:', friends)


def cube(x):
    return x * x * x

def square(x):
    return x * x
def add(x, y):
    return x + y

cube_result = cube(3)
square_result = square(4)
add_result = add(5, 6)  
print('Cube of 3:', cube_result)


###game to guess a word
# secret_word = 'python'
# out_of_guesses = False
# guesses = ''
# guess_limit = 0

# while not out_of_guesses:
#     if guess_limit > 3:
#         print('You have run out of guesses')
#         break
#     if len(guesses) <= 3:
#         guess = input('Guess a letter: ')
#         guess_limit += 1
#         if guess in secret_word:
#             print('Good job! you win.')
#         else:
#             print('you can try again')
            
# create a x^y
def power(x, y):
    return x ** y
print('2^3 =', power(3, 4))


from student import Student

student1 = Student('John', 20)
student1.display_info()
print('Student Name:', student1.name)


# smash candies
def to_smash(total_candies, num_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between friends.
    
    If num_friends is not provided, assumes 3 friends as default.
    
    >>> to_smash(91)
    1
    >>> to_smash(91, 3)
    1
    >>> to_smash(10, 4)
    2
    """
    return total_candies % num_friends

# Test the function
print("Candies to smash with 91 total (3 friends):", to_smash(91))
print("Candies to smash with 91 total (4 friends):", to_smash(91, 4))
print("Candies to smash with 10 total (4 friends):", to_smash(10, 4))

# Round to two decimal places function
def round_to_two_places(number):
    """Round a number to two decimal places.
    
    >>> round_to_two_places(9.9999)
    10.0
    >>> round_to_two_places(3.14159)
    3.14
    """
    return round(number, 2)

# Test the round function
print("9.9999 rounded to 2 places:", round_to_two_places(9.9999))
print("3.14159 rounded to 2 places:", round_to_two_places(3.14159))
print("2.675 rounded to 2 places:", round_to_two_places(2.675))

# Absolute value examples
print("\nAbsolute value examples:")
print("abs(-5):", abs(-5))
print("abs(5):", abs(5))
print("abs(-3.14):", abs(-3.14))
print("abs(10 - 3):", abs(10 - 3))  # Absolute difference
print("abs(3 - 10):", abs(3 - 10))  # Absolute difference

# Sign function - returns 1 for positive, -1 for negative, 0 for zero
def sign(num):
    """Return the sign of a number.
    
    Returns 1 for positive numbers, -1 for negative numbers, and 0 for zero.
    
    >>> sign(5)
    1
    >>> sign(-3)
    -1
    >>> sign(0)
    0
    """
    if num > 0:
        return 1
    elif num < 0:
        return -1
    else:
        return 0

# Test the sign function
print("\nSign function examples:")
print("sign(5):", sign(5))
print("sign(-3):", sign(-3))
print("sign(0):", sign(0))
print("sign(3.14):", sign(3.14))
print("sign(-2.7):", sign(-2.7))

# Verbose is_negative function
def is_negative(number):
    if number < 0:
        return True
    else:
        return False

# Concise version - reduces 4 lines to 1 line!
def concise_is_negative(number):
    return number < 0


# Alternative more concise version
def select_second_concise(L):
    """Concise version using conditional expression."""
    return L[1] if len(L) >= 2 else None

def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    return any([num % 7 == 0 for num in nums])

# Element-wise comparison function
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    >>> elementwise_greater_than([5, 1, 8, 2], 3)
    [True, False, True, False]
    """
    # Method 1: Using list comprehension (most Pythonic)
    return [value > thresh for value in L]

# Alternative implementations
def elementwise_greater_than_loop(L, thresh):
    """Alternative implementation using a for loop."""
    result = []
    for value in L:
        if value > thresh:
            result.append(True)
        else:
            result.append(False)
    return result

def elementwise_greater_than_map(L, thresh):
    """Alternative implementation using map."""
    return list(map(lambda x: x > thresh, L))

# Test the elementwise_greater_than functions
print("\nTesting elementwise_greater_than functions:")
test_list = [1, 2, 3, 4, 5]
threshold = 2

print(f"List: {test_list}, Threshold: {threshold}")
print("elementwise_greater_than:", elementwise_greater_than(test_list, threshold))
print("elementwise_greater_than_loop:", elementwise_greater_than_loop(test_list, threshold))
print("elementwise_greater_than_map:", elementwise_greater_than_map(test_list, threshold))

# More test cases
print(f"\nelementwise_greater_than([5, 1, 8, 2], 3): {elementwise_greater_than([5, 1, 8, 2], 3)}")
print(f"elementwise_greater_than([0, 0, 0], 1): {elementwise_greater_than([0, 0, 0], 1)}")
print(f"elementwise_greater_than([10, 20, 30], 15): {elementwise_greater_than([10, 20, 30], 15)}")

# Menu is boring function
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    
    >>> menu_is_boring(['pizza', 'pasta', 'pizza', 'salad'])
    False
    >>> menu_is_boring(['pizza', 'pizza', 'pasta', 'salad'])
    True
    >>> menu_is_boring(['pasta', 'salad', 'pasta', 'pasta'])
    True
    >>> menu_is_boring(['pizza'])
    False
    >>> menu_is_boring([])
    False
    """
    # Check each consecutive pair of meals
    for i in range(len(meals) - 1):
        if meals[i] == meals[i + 1]:
            return True
    return False

# Alternative implementation using zip
def menu_is_boring_zip(meals):
    """Alternative implementation using zip to compare consecutive elements."""
    return any(meal1 == meal2 for meal1, meal2 in zip(meals, meals[1:]))

# Alternative implementation using enumerate
def menu_is_boring_enumerate(meals):
    """Alternative implementation using enumerate."""
    for i, meal in enumerate(meals[1:], 1):
        if meal == meals[i - 1]:
            return True
    return False

# Test the menu_is_boring functions
print("\nTesting menu_is_boring functions:")

test_menus = [
    ['pizza', 'pasta', 'pizza', 'salad'],      # No consecutive duplicates
    ['pizza', 'pizza', 'pasta', 'salad'],      # Pizza repeated
    ['pasta', 'salad', 'pasta', 'pasta'],      # Pasta repeated at end
    ['burger', 'soup', 'burger', 'soup'],      # No consecutive duplicates
    ['salad'],                                  # Single meal
    [],                                         # Empty menu
    ['pizza', 'pizza', 'pizza']                # All same meal
]

for menu in test_menus:
    result = menu_is_boring(menu)
    result_zip = menu_is_boring_zip(menu)
    result_enum = menu_is_boring_enumerate(menu)
    print(f"menu_is_boring({menu}): {result} (zip: {result_zip}, enum: {result_enum})")

