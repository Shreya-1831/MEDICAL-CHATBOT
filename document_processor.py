import os
from typing import List, Dict
import PyPDF2
import pytesseract
from PIL import Image

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(self, file_path: str, filename: str) -> List[Dict[str, str]]:
        """Process a document and return text chunks"""
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_ext == '.txt':
            text = self.extract_text_from_txt(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            text = self.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Split into chunks
        chunks = self.chunk_text(text, filename)
        return chunks
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
        
        return text
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")
    
    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            raise Exception(f"Error performing OCR on image: {str(e)}")
    
    def chunk_text(self, text: str, source: str) -> List[Dict[str, str]]:
        """Split text into chunks with overlap"""
        chunks = []
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Split into chunks
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Find the last space within the chunk to avoid cutting words
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': chunk_id
                })
                chunk_id += 1
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text) - self.chunk_overlap:
                break
        
        return chunks
    
    def recursive_character_split(self, text: str, source: str) -> List[Dict[str, str]]:
        """Alternative chunking method using recursive character splitting"""
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        
        def split_text(text, separators):
            if len(text) <= self.chunk_size:
                return [text]
            
            separator = separators[0] if separators else ""
            parts = text.split(separator) if separator else [text]
            
            result = []
            current_chunk = ""
            
            for part in parts:
                if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                    current_chunk += (separator if current_chunk else "") + part
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    
                    if len(part) > self.chunk_size and len(separators) > 1:
                        result.extend(split_text(part, separators[1:]))
                    else:
                        current_chunk = part
            
            if current_chunk:
                result.append(current_chunk)
            
            return result
        
        text_chunks = split_text(text, separators)
        
        for idx, chunk in enumerate(text_chunks):
            if chunk.strip():
                chunks.append({
                    'text': chunk.strip(),
                    'source': source,
                    'chunk_id': idx
                })
        
        return chunks
