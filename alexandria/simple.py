import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Load .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

def ask_gemini_about_docs(documents, question):
    context = "\n".join(documents)

    # Prompt
    prompt_template = """
You are an assistant who reads the following documents:

{context}

Answer this question based on the documents:

{question}
"""
    prompt = PromptTemplate.from_template(prompt_template)
    input_text = prompt.format(context=context, question=question)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=input_text)
    ]

    # Use .invoke() instead of deprecated .run() or __call__()
    response = llm.invoke(messages)
    return response.content

# Example usage
docs = [
    "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "LangChain is a Python framework for building applications using LLMs."
]

question = "Who created Python and what is LangChain used for?"
answer = ask_gemini_about_docs(docs, question)
print(answer)
