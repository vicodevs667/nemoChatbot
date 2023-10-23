from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import os
import sys
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA

#prompt_template =
"""
You are an AI coding assistant and your task to solve the coding problems, and return coding snippets based on the
Query: {query}

You just return helpful answer and nothing else
Helpful Answer: (escribe como mexicano informal)
"""

prompt_template = """
You are an AI tourism assistant to visit Mexico, and return recommendations about places to visit in Mexico
Query: {query}

Helpful Answer (escribe como colombiano informal):
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['query'])

llm = CTransformers(model="model/codellama-7b-instruct.ggmlv3.Q4_0.bin",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.2
                    )
llm_chain = LLMChain(prompt=prompt, llm=llm)


#llm_response = llm_chain.run({"query": "Write a python code to load a CSV file using pandas library"})
#llm_response = llm_chain.run({"query": "Qué lugares puedo visitar en México"})

#print(llm_response)

loader = PyPDFDirectoryLoader("data")
data = loader.load()
print(data)