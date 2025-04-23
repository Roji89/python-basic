message = "Hello, World!"

# Replaces "World" with "universe" in the message string
newMessage = message.replace("World", "universe")

print(newMessage)

greeting = "Coucou"
name = "Roja"

# Creates a formatted string combining greeting and name
greetingMessage = f"{greeting} {name} !"
print(greetingMessage)

courses = ["Python", "Java", "C++", "Nodejs"]

# Prints the number of elements in the courses list
print(len(courses))

# Prints the first two elements of the courses list
print(courses[0:2])

# Inserts "JavaScript" at the beginning of the courses list
print(courses.insert(0, "JavaScript"))
print(courses)

# Removes and returns the last element of the courses list
popped = courses.pop()
print(popped)

# Reverses the order of elements in the courses list
courses.reverse()
print(courses)  # Print the modified list

# Sorts the courses list in ascending order
courses.sort()
print(courses)  # Print the modified list

numbers = [1, 8, 3, 6, 5]
# Sorts the numbers list in ascending order
numbers.sort()
print(numbers)  # Print the modified list

numbers.sort(reverse=True)
print(numbers)  # Print the modified list

# Creates a list of tuples with names and ages
print("Java" in courses)

for cours in courses:
    print(cours)

for index, cours in enumerate(courses, start=1):
    print(index, cours)
course_string = ", ".join(courses)
print(course_string)  # Print the joined string

new_list = course_string.split("- ")
print(new_list)  # Print the split list

# List comprehension: Create a new list with squares of numbers from 1 to 5
squares = [x**2 for x in range(1, 6)]
print("Squares:", squares)

# Dictionary: Create a dictionary and demonstrate basic operations
person = {"name": "Roja", "age": 25, "city": "Paris"}
print("Person dictionary:", person)
print("Name:", person["name"])  # Access a value by key
person["age"] = 26  # Update a value
print("Updated age:", person["age"])
person["profession"] = "Engineer"  # Add a new key-value pair
print("Updated dictionary:", person)

# Set: Demonstrate set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print("Union:", set1 | set2)  # Union of sets
print("Intersection:", set1 & set2)  # Intersection of sets
print("Difference:", set1 - set2)  # Difference of sets

# Using zip to combine two lists into a list of tuples
names = ["Alice", "Bob", "Charlie"]
ages = [24, 27, 22]
combined = list(zip(names, ages))
print("Combined list of tuples:", combined)

# Using a lambda function to sort a list of tuples by the second element
sorted_combined = sorted(combined, key=lambda x: x[1])
print("Sorted by age:", sorted_combined)

# Demonstrating exception handling
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("Error:", e)
finally:
    print("This block always executes.")

# File handling: Writing to and reading from a file
with open("example.txt", "w") as file:
    file.write("Hello, this is a test file.\n")
    file.write("Python is fun!")

with open("example.txt", "r") as file:
    content = file.read()
    print("File content:\n", content)

