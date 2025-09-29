'''Test script for the Alexandria library using my thesis.'''
import pypdf
from alexandria.library import Book
from alexandria.librarian import Librarian

# Decalre and initialize the librarian
librarian = Librarian('Demetrius of Phalerum', debug=False)

# Load in my thesis from local file
byates_thesis = pypdf.PdfReader('data/bryates_thesis.pdf')
# Extract the text from each page
text = []
for page in byates_thesis.pages:
    text.append(page.extract_text())

# Declare the book
byates = Book('Measurement of the Shape of the b Quark Fragmentation Function Using Charmed Mesons in Proton-Proton Collisions at a Center of Mass-Energy of 13 TeV', author='Brent R. Yates', doc=text, category='thesis', doc_type='pdf')
print(byates)

# Add the book to the librarian's collection
librarian.add_book(byates)
# print(librarian)
print(librarian.list_books())
# Query the book summaries
print(librarian.query_book_summaries(byates))
