from pypdf import PdfReader

reader = PdfReader("data/1706.03762v7.pdf")

print("Pages:", len(reader.pages))

text = reader.pages[0].extract_text()

print(text[:1000])