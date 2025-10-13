# AI Model & Prompt Engineering Consulting Guide
# Complete framework for client recommendations

class AIConsultingFramework:
    def __init__(self):
        self.model_recommendations = {
            "startup": {
                "budget": "Low",
                "models": ["Llama-3-8B", "Granite-3-8B", "Mixtral-8x7B"],
                "deployment": "Cloud API",
                "cost_per_1k_tokens": 0.001
            },
            "mid_market": {
                "budget": "Medium", 
                "models": ["GPT-4-turbo", "Claude-3-Sonnet", "Llama-3-70B"],
                "deployment": "Hybrid",
                "cost_per_1k_tokens": 0.01
            },
            "enterprise": {
                "budget": "High",
                "models": ["GPT-4", "Claude-3-Opus", "Custom fine-tuned"],
                "deployment": "Private cloud/On-premise",
                "cost_per_1k_tokens": 0.06
            }
        }
        
        self.prompt_strategies = {
            "simple_tasks": {
                "method": "Zero-shot",
                "template": "Direct instruction + clear format",
                "examples": ["Classification", "Simple Q&A", "Basic extraction"]
            },
            "complex_reasoning": {
                "method": "Chain-of-Thought",
                "template": "Step-by-step reasoning",
                "examples": ["Financial analysis", "Technical troubleshooting", "Strategic planning"]
            },
            "domain_expertise": {
                "method": "Role-based + Few-shot",
                "template": "Expert persona + examples",
                "examples": ["Legal advice", "Medical consultation", "Engineering analysis"]
            },
            "creative_tasks": {
                "method": "Creative prompting",
                "template": "Context + style + constraints",
                "examples": ["Content creation", "Marketing copy", "Creative writing"]
            }
        }
    
    def recommend_model(self, client_type, use_case, requirements):
        """
        Provide model recommendation based on client profile
        """
        base_rec = self.model_recommendations.get(client_type, self.model_recommendations["mid_market"])
        
        # Adjust based on requirements
        if requirements.get("accuracy") == "critical":
            base_rec["models"] = [model for model in base_rec["models"] if "GPT-4" in model or "Claude-3" in model]
        
        if requirements.get("latency") == "real_time":
            base_rec["models"] = [model for model in base_rec["models"] if "8B" in model or "turbo" in model]
            
        return base_rec
    
    def recommend_prompt_strategy(self, task_complexity, accuracy_needed, examples_available):
        """
        Recommend optimal prompting approach
        """
        if task_complexity == "simple" and accuracy_needed != "critical":
            return self.prompt_strategies["simple_tasks"]
        elif task_complexity == "complex" or accuracy_needed == "critical":
            return self.prompt_strategies["complex_reasoning"]
        elif examples_available and accuracy_needed == "high":
            return self.prompt_strategies["domain_expertise"]
        else:
            return self.prompt_strategies["simple_tasks"]

# Example usage for client consultation
def client_consultation_example():
    consultant = AIConsultingFramework()
    
    # Client scenario
    client_profile = {
        "type": "mid_market",
        "use_case": "customer_support_automation",
        "requirements": {
            "accuracy": "high",
            "latency": "real_time",
            "volume": "1000_requests_per_day"
        }
    }
    
    # Get recommendations
    model_rec = consultant.recommend_model(
        client_profile["type"],
        client_profile["use_case"], 
        client_profile["requirements"]
    )
    
    prompt_rec = consultant.recommend_prompt_strategy(
        task_complexity="medium",
        accuracy_needed="high", 
        examples_available=True
    )
    
    print("=== CLIENT RECOMMENDATION REPORT ===")
    print(f"Recommended Models: {model_rec['models']}")
    print(f"Deployment: {model_rec['deployment']}")
    print(f"Estimated Cost: ${model_rec['cost_per_1k_tokens']} per 1K tokens")
    print(f"Prompt Strategy: {prompt_rec['method']}")
    print(f"Use Cases: {prompt_rec['examples']}")

if __name__ == "__main__":
    client_consultation_example()
