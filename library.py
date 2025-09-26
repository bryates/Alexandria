class Book():
    def __init__(self, title, doc):
        self.title = title
        self.tokenize(doc)

    def tokenize(self, doc):
        self.token = doc.strip().split()

    def __str__(self):
        return f"Title: '{self.title}'\nToken: {self.token}"

    def __repr__(self):
        return f"Book('{self.title}', {self.token})"
