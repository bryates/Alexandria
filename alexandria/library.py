class Book():
    def __init__(self, title, author, doc, category, doc_type='pdf'):
        self.title = title
        self.author = author
        self.tokenize(doc)
        self.category = category
        self.type = doc_type

    def tokenize(self, doc):
        if isinstance(doc, list):
            self.token = []
            for d in doc:
                self.token.extend(d.strip().split())
        else:
            self.token = doc.strip().split()

    def __str__(self):
        return f"Title: '{self.title}'\nAuthor: '{self.author}'\nBook type: '{self.category}'\nDocument type: '{self.type}'"

    def __repr__(self):
        return f"Book('{self.title}', {self.author}', '{self.category}', '{self.type}')"
