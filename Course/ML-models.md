# ML models:
1. Supervised Learning model:
  give an example to machine
   - Linear Regression
   - Logistic Regression
   - Decision Trees
   - Random Forests
   - Support Vector Machines (SVM)
   - Neural Networks
2. Unsupervised Learning model:
  dont give any example to machine
   - K-Means Clustering
   - Hierarchical Clustering
   - Principal Component Analysis (PCA)
   - t-Distributed Stochastic Neighbor Embedding (t-SNE)
   - Autoencoders
3.semi-supervised Learning model:
  mix of supervised and unsupervised learning
   - Semi-supervised Support Vector Machines
   - Semi-supervised Neural Networks
   - Graph-based methods
   - Self-training methods
   - Co-training methods

![alt text](image.png)

---

## Machine Learning Workflow/Lifecycle

The ML process follows a circular workflow with interconnected phases:

```
                    ┌─────────────────────┐
                    │   Problem           │
                    │   Definition        │
                    │        ?            │
                    └──────────┬──────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌─────────────────────┐     ┌─────────────────────┐
    │   Model             │     │   Data              │
    │   Deployment        │     │   Collection        │
    │        📊           │     │        📋           │
    └──────────┬──────────┘     └──────────┬──────────┘
               │                           │
               ▲                           ▼
    ┌─────────────────────┐     ┌─────────────────────┐
    │   Model Development │     │   Data              │
    │   and Evaluation    │     │   Preparation       │
    │        ⚙️           │     │        🔄           │
    └─────────────────────┘     └─────────────────────┘
               ▲                           │
               └───────────────────────────┘
```
prescriptive => decision optimisation /recomendation engine
generative => assistants / chatbots 

use cases => data => models(fundation models) => process(PEFT ,RAG , AGENT) => embebded
## 7 ML Tools
📊 Data Processing and Analytics =>pandas,numpy, posgresql,hadoop,spark,apache kafka
📈 Data Visualization =>Matplotlib, Tableau, Power BI
🤖 Machine Learning =>Scikit-learn, XGBoost
🧠 Deep Learning =>TensorFlow, PyTorch
👁️ Computer Vision =>OpenCV, YOLO, ResNet
📝 Natural Language Processing =>NLTK, spaCy, Transformers
🎨 Generative AI =>GPT-4, DALL-E, GitHub Copilot

apache kafak+pandas+numpy+seaborn+ggplot2+
tensorflow+
opencv+scikit-image+torchVision
NLTK+TExtBlob+stanza
hugging face transformer => tsx,language,translation,sentiment analysis
Dall-E => generate image from text
Stable Diffusion => generate image from text

---

## When to Use Different ML Techniques

**Classification**: When you need to categorize new data (spam detection, medical diagnosis)

**Clustering**: When you want to explore data structure (customer segmentation, data exploration)

**Regression**:  Predict continuous numerical values (prices, temperatures, quantities)
- *Simple Regression*: Single feature predicts continuous value (CO2 emission from engine size)
- *Multiple Regression*: Multiple features predict continuous value (house price from size, bedrooms, location)
* Multicollinearity: When features are too similar, it confuses the model
* Overfitting: Too many similar features can make model too complex
* Efficiency: Fewer features = faster training and prediction

🔹 Simple Linear Regression
Uses: 1 feature (independent variable)
Formula: ŷ = θ₀ + θ₁x₁
Geometry: Creates a line in 2D space
Example: Predict CO2 emissions using only engine size
🔹 Multiple Linear Regression
Uses: 2+ features (independent variables)
Formula: ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
Geometry: Creates a plane (2 features) or hyperplane (3+ features)
Example: Predict CO2 emissions using engine size AND fuel efficiency

**Association**: When you want to find patterns in purchases or behaviors (recommendation systems, market analysis) 