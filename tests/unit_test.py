from library import Book

b = Book('hi', 'hello there\nhow are you?')
def test_book():
    assert b.title == b.title


def test_token():
    assert b.token == ['hello', 'there', 'how', 'are', 'you?']
