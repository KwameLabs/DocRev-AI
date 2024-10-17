# importing required modules
from dotenv import load_dotenv
from pypdf import PdfReader

# creating a pdf reader object
reader = PdfReader('app\pdfs\Guidelines_for_Public_Policy_Formulation_in_Ghana_Final_Nov20201_ML.pdf')

# printing number of pages in pdf file
print(len(reader.pages))

# getting a specific page from the pdf file
page = reader.pages[0]

# extracting text from page
text = page.extract_text()
print(text)

def main():
    load_dotenv()
#tests whether the file is being run directly or imported
if __name__ == '__main__':
    main()