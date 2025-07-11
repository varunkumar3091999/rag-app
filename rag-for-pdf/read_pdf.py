import pdfplumber
import re

def extract_text_from_pdf(pdf_path):
  all_text = ''
  
  with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
      page_text = page.extract_text()
      if(page_text):
        all_text += page_text + "\n"
    return all_text
  
def clean_text(text):
  # replace multiple new lines into single new line
  text = re.sub(r'\n+','\n', text)
  # remove extra spaces
  text = re.sub(r' +', ' ', text)
  text = text.replace("\n", " ")
  return text.strip()

def split_into_chunks(text, chunk_size=1000, overlap=0):
  chunks = []
  start = 0
  while start < len(text):
    end  = start + chunk_size
    chunk = text[start:end]
    chunks.append(chunk)
    start += chunk_size + overlap
  return chunks

def remove_header_and_footer(text, keywords):
  cleaned_text = text
  for keyword in keywords:
    cleaned_text = cleaned_text.replace(keyword, "")
  return cleaned_text
