import fitz  # for reading and interacting with pdf


# function to read text from .txt file
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
# function to read a pdf and save it as a new file
def read_pdf(file_path):
    pdf_doc = fitz.open(file_path)
    text = ""
    for page_number in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_number)
        text += page.get_text()
    return text

def save_extracted_text_to_file(text, file_path):
    with open(file_path + '.txt', 'w', encoding='utf-8') as file:
        file.write(text)