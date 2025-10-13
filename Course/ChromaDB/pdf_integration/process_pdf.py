
from langchain.document_loaders import PyPDFLoader
import pdfplumber

def process_pdf(pdf_path):
  # Process the PDF file and extract relevant information
  # Return a dictionary of the extracted information
  return



def _extract_pdf_content(pdf_path):
  text_content =""
  pdf_metadata={}
  try:
    with pdfplumber.open(pdf_path) as pdf:
      pdf_metadata = pdf.metadata
      for page in pdf.pages:
        text_content += page.extract_text() + "\n"
  except Exception as e:
    print(f"Error processing PDF: {str(e)}")
  return 