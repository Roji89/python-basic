def llm_model(prompt_txt, params=None):
    
    model_id = "ibm/granite-3-2-8b-instruct"

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    if params:
        default_params.update(params)

    # Set up credentials for WatsonxLLM
    url = "https://us-south.ml.cloud.ibm.com"
    api_key = "your api key here"
    project_id = "skills-network"

    credentials = {
        "url": url,
        # "api_key": api_key
        # uncomment the field above and replace the api_key with your actual Watsonx API key
    }
    
    # Create LLM directly
    granite_llm = WatsonxLLM(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=default_params
    )
    
    response = granite_llm.invoke(prompt_txt)
    return response



params = {
    "max_new_tokens": 128, # Try 256 or 512 for more detailed answers
    "min_new_tokens": 10, # Increase to 25-50 if you want more substantial answers
    "temperature": 0.5, # Controls randomness in generation (0.0-1.0)
                       # Lower (0.1-0.3): More focused, consistent, factual responses
                      # Higher (0.7-1.0): More creative, diverse, unpredictable outputs
    "top_p": 0.2, # Nucleus sampling - considers only highest probability tokens
                       # Lower values (0.1-0.3): More conservative, focused text
                       # Higher values (0.7-0.9): More diverse vocabulary and ideas
    "top_k": 1 # Limits token selection to top k most likely tokens
                       # 1 = greedy decoding (always picks most likely token)
                       # Try 40-50 for more varied outputs
}

prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The benefits of sustainable energy include"
]

response = llm_model(prompts, params)
for prompt in prompts:
    response = llm_model(prompt, params)
    print(f"prompt: {prompt}\n")
    print(f"response : {response}\n")


## THE USEFUL PHRASES:  
### consider this as a situation : .... answer: ...
### subject : ... answer : ....




# LCEL
model_id = "meta-llama/llama-3-3-70b-instruct"

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
}

url = "https://us-south.ml.cloud.ibm.com"
project_id = "skills-network"

llm = WatsonxLLM(
        model_id=model_id,
        url=url,
        project_id=project_id,
        params=parameters
    )
llm

template = """Tell me a {adjective} joke about {content}.
"""
prompt = PromptTemplate.from_template(template)
prompt.format(adjective="funny", content="chickens")
from langchain_core.runnables import RunnableLambda

def format_prompt(variables):
    return prompt.format(**variables)

joke_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

response = joke_chain.invoke({"adjective": "funny", "content": "chickens"})
print(response)

# Summerization example
content = """
    The rapid advancement of technology in the 21st century has transformed various industries, including healthcare, education, and transportation. 
    Innovations such as artificial intelligence, machine learning, and the Internet of Things have revolutionized how we approach everyday tasks and complex problems. 
    For instance, AI-powered diagnostic tools are improving the accuracy and speed of medical diagnoses, while smart transportation systems are making cities more efficient and reducing traffic congestion. 
    Moreover, online learning platforms are making education more accessible to people around the world, breaking down geographical and financial barriers. 
    These technological developments are not only enhancing productivity but also contributing to a more interconnected and informed society.
"""

template = """Summarize the {content} in one sentence.
"""
prompt = PromptTemplate.from_template(template)

summarize_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

summary = summarize_chain.invoke({"content": content})
print(summary)


# Question Answering example
content = """
    The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets. 
    The inner planets—Mercury, Venus, Earth, and Mars—are rocky and solid. 
    The outer planets—Jupiter, Saturn, Uranus, and Neptune—are much larger and gaseous.
"""

question = "Which planets in the solar system are rocky and solid?"

template = """
    Answer the {question} based on the {content}.
    Respond "Unsure about answer" if not sure about the answer.
    
    Answer:
    
"""
prompt = PromptTemplate.from_template(template)

# Create the LCEL chain
qa_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)

# Run the chain
answer = qa_chain.invoke({"question": question, "content": content})
print(answer)

# Code generation

description = """
    Retrieve the names and email addresses of all customers from the 'customers' table who have made a purchase in the last 30 days. 
    The table 'purchases' contains a column 'purchase_date'
"""

template = """
    Generate an SQL query based on the {description}
    
    SQL Query:
    
"""
prompt = PromptTemplate.from_template(template)

# Create the LCEL chain
sql_generation_chain = (
    RunnableLambda(format_prompt) 
    | llm 
    | StrOutputParser()
)

# Run the chain
sql_query = sql_generation_chain.invoke({"description": description})
print(sql_query)



# Role base conversation
role = """
    Dungeon & Dragons game master
"""

tone = "engaging and immersive"

template = """
    You are an expert {role}. I have this question {question}. I would like our conversation to be {tone}.
    
    Answer:
    
"""
prompt = PromptTemplate.from_template(template)


roleplay_chain = (
    RunnableLambda(format_prompt)
    | llm 
    | StrOutputParser()
)


while True:
    query = input("Question: ")
    
    if query.lower() in ["quit", "exit", "bye"]:
        print("Answer: Goodbye!")
        break
        
    response = roleplay_chain.invoke({"role": role, "question": query, "tone": tone})
    print("Answer: ", response)


### Complete example ###

## Starter code: provide your solutions in the TODO parts
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# First initialize your LLM
model_id = "meta-llama/llama-3-3-70b-instruct" ## Or you can use other LLMs available via watsonx.ai

# Use these parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 512,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.2, # this randomness or creativity of the model's responses
}

url = "https://us-south.ml.cloud.ibm.com"
project_id = "skills-network"

# TODO: Initialize your LLM
llm = WatsonxLLM(
        model_id=model_id,
        url=url,
        project_id=project_id,
        params=parameters
    )
llm


# Here is an example template you can use
template = """
Analyze the following product review:
"{review}"

Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)
"""

# TODO: Create your prompt template
product_review_prompt = PromptTemplate.from_template(template)


# TODO: Create a formatting function
def format_review_prompt(variables):
    return product_review_prompt.format(**variables)
    pass

# TODO: Build your LCEL chain
review_analysis_chain = (
    RunnableLambda(format_review_prompt)
    | llm 
    | StrOutputParser()
)

# Example reviews to process
reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful."
]

# TODO: Process the reviews
for review in reviews:
    response= review_analysis_chain()
    print("Answer: ", response)
    pass

### Smarter AI app ###

    # !pip install --force-reinstall --no-cache-dir tenacity==8.2.3 --user
    # !pip install "ibm-watsonx-ai==1.0.8" --user
    # !pip install "ibm-watson-machine-learning==1.0.367" --user
    # !pip install "langchain-ibm==0.1.7" --user
    # !pip install "langchain-community==0.2.10" --user
    # !pip install "langchain-experimental==0.0.62" --user
    # !pip install "langchainhub==0.1.18" --user
    # !pip install "langchain==0.2.11" --user
    # !pip install "pypdf==4.2.0" --user
    # !pip install "chromadb==0.4.24" --user

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

