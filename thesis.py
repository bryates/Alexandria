'''Test script for the Alexandria library using my thesis.'''
import pypdf
from alexandria.library import Book
from alexandria.librarian import Librarian

librarian = Librarian('Demetrius of Phalerum', debug=False)

byates_thesis = pypdf.PdfReader('data/bryates_thesis.pdf')
# byates_thesis = pypdf.PdfReader(r"C:\Users\yates.313\Alexandria\data\bryates_thesis.pdf")
text = []
for page in byates_thesis.pages:
    text.append(page.extract_text())
# byates = Book('B.R. Yates Thesis', author='Brent Yates', doc='https://escholarship.org/content/qt372784x8/qt372784x8.pdf', category='thesis', doc_type='pdf')
byates = Book('Measurement of the Shape of the b Quark Fragmentation Function Using Charmed Mesons in Proton-Proton Collisions at a Center of Mass-Energy of 13 TeV', author='Brent R. Yates', doc=text, category='thesis', doc_type='pdf')
print(byates)

librarian.add_book(byates)
# print(librarian)
print(librarian.list_books())
print(librarian.query_book_summaries(byates))
