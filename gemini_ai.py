from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini (free tier available)
# Get your free API key from: https://makersuite.google.com/app/apikey
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def get_completion_gemini(prompt):
    try:
        
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to:")
        print("1. Get free API key from: https://makersuite.google.com/app/apikey")
        print("2. Add GEMINI_API_KEY to your .env file")
        return None

# Your existing prompt
prompt = """
Your task is to determine if the student's solution is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem including the final total. 
- Then compare your solution to the student's solution and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until you have done the problem yourself.

Use the following format:
Question:
```
question here
```
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```

Question:
```
I'm building a solar power installation and I need help working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost me a flat $100k per year, and an additional $10 / square foot
What is the total cost for the first year of operations as a function of the number of square feet.
``` 
Student's solution:
```
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
```
Actual solution:
"""

print("🚀 Using Google Gemini (Free!)")
response = get_completion_gemini(prompt)
if response:
    print("\n" + "="*50)
    print("🤖 Gemini Response:")
    print("="*50)
    print(response)
