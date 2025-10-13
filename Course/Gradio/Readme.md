pip install virtualenv 
virtualenv my_env # create a virtual environment named my_env
source my_env/bin/activate # activate my_env
# installing necessary pacakges in my_env
python3 -m pip install \
gradio==4.44.0 \
pydantic==2.10.6 \
ibm-watsonx-ai==1.1.2 \
langchain==0.2.11 \
langchain-community==0.2.10 \
langchain-ibm==0.1.11


python3.11 gradio_demo.py

