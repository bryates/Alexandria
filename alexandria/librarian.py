'''Module for managing a collection of books in the Alexandria library.'''
'''Example code to call Gemini API using the Google Generative AI SDK in Python.'''
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# Load .env
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    raise ValueError('Please set GEMINI_API_KEY in .env')

# Configure the SDK
genai.configure(api_key=api_key)

class Librarian:
    '''Initialize the librarian with an empty book collection.'''
    def __init__(self, name, debug=False):
        self.name = name
        self.books = []
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # or "gemini-2.5-pro", etc.
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7
        )
        self.debug = debug

        print(f"Hello, my name is {self.name}, and I'll be your librarian.")

    def add_book(self, book):
        '''Add a book to the collection.'''
        self.books.append(book)
        print(f"I've added \"{book.title}\" by {book.author} to the collection.")

    def list_books(self):
        '''List all book titles in the collection.'''
        books = [f'"{book.title}" by {book.author}' for book in self.books]
        return 'Collection: \n\t' + '\n\t'.join(books)

    def query_book_summaries(self, query=None):
        '''Use Gemini to answer a question based on book summaries.'''
        if self.debug:
            print(f'{self.books=}')
        summaries = ['\n'.join(book.token) for book in self.books]
        if self.debug:
            print(f'{summaries=}')
        if not query:
            question = '\n'.join(summaries)
        else:
            #question = f'{query.title} by {query.author}, {query.category}, {query.type}'
            question = str(query.token)

        prompt = '''You are an assistant who reads the following PDF documents. Pay particular attention to any sections that cover measrument, results, and conclusion. Give a nice bullet breakdown of the key points.'''
        # Simple message call
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f'Write a summary of {question}.')
        ]
        if self.debug:
            print('Prompt:', prompt)
            print('Messages:', messages)

        response = self.llm.invoke(messages)
        return response.content

    def __str__(self):
        return f'Librarian {self.name} with {len(self.books)} books'

    def __repr__(self):
        return self.__str__()
