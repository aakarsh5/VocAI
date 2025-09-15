# llm.py
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# HuggingFace model locally
generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

llm = HuggingFacePipeline(pipeline=generator)

# prompt template
template = """You are a helpful AI voice assistant. Answer the user query concisely.
User: {question}
Assistant:"""
prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)

def get_llm_response(prompt: str) -> str:
    return chain.run(question=prompt)
