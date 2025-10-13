# Import necessary packages
import gradio as gr

# Mock LLM function for demonstration (since we don't have Watsonx credentials)
def mock_llm_response(prompt):
    """
    Mock LLM that provides sample responses.
    In the real implementation, this would be connected to IBM Watsonx AI.
    """
    responses = {
        "hello": "Hello! How can I help you today?",
        "how are you": "I'm doing well, thank you for asking! How can I assist you?",
        "what is ai": "Artificial Intelligence (AI) is a branch of computer science that aims to create machines that can perform tasks that typically require human intelligence.",
        "python": "Python is a high-level programming language known for its simplicity and versatility. It's widely used in AI, web development, and data analysis.",
        "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from data without being explicitly programmed for every task."
    }
    
    prompt_lower = prompt.lower()
    
    # Check for keyword matches
    for key, response in responses.items():
        if key in prompt_lower:
            return f"🤖 {response}\n\n(This is a demo response. In production, this would use IBM Watsonx AI.)"
    
    # Default response
    return f"🤖 Thank you for your question: '{prompt}'\n\nI'm a demo chatbot. In the real implementation, I would be powered by IBM Watsonx AI's Mixtral model to provide intelligent responses.\n\nTry asking about: hello, AI, Python, or machine learning!"

# Function to generate a response from the model
def generate_response(prompt_txt):
    if not prompt_txt or prompt_txt.strip() == "":
        return "Please enter a question or message."
    
    # In the real implementation, this would call:
    # generated_response = watsonx_llm.invoke(prompt_txt)
    # return generated_response
    
    return mock_llm_response(prompt_txt)

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port= 7860)