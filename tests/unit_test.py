'''Unit tests for the Book class in library.py'''

import pytest
from alexandria.library import Book

@pytest.fixture(scope='module', autouse=True)
def b():
    '''Fixture that returns a Book instance for testing'''
    return Book('hi', 'hello there\nhow are you?', doc='hello there how are you?', category='test', doc_type='txt')


def test_book(b):
    '''Test the Book class'''
    assert b.title == 'hi'


def test_token(b):
    '''Test the tokenization of the Book class'''
    assert b.token == ['hello', 'there', 'how', 'are', 'you?']
