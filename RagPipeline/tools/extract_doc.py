import pymupdf4llm
import re 
import unicodedata
import os 
from tools.clean_text import clean_text
from tools.clean_markdown import clean_markdown
import json 
from bs4 import BeautifulSoup
import markdown

class ExtractDocContent:
    """_summary_
    * **Extract the content on pdf page-by-page with word len ,char len,page number **
    
    
    """
    def __init__(self,doc_url) -> None:
        super().__init__()
        self.doc_url= doc_url
    
    
    def extract_pdf(self,path):
        """_summary_

        Args:
            path (str): path of the pdf 

        Returns:
            _type_: _description_
        """
        pdf_text = pymupdf4llm.to_markdown(
            path
        )
        pdf_text = clean_text(pdf_text)
        pdf_text = clean_markdown(pdf_text)
        return pdf_text
    
    def extract_html(self,path):
        """_summary_

        Args:
            path (str): the path of the html 
        """
        with open(path,"r",encoding="utf-8") as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        text = clean_text(text)
        text = clean_markdown(text)
        
        return text

        
    def extract_markdown(self,path):
        """_summary_

        Args:
            path (str): path of the markdown file 
        """
        with open(path,"r",encoding="utf-8") as f:
            md_content = f.read()
        
        html_out = markdown.markdown(md_content)
        
        soup = BeautifulSoup(html_out,"html.parser")
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        text = clean_text(text)
        text = clean_markdown(text)
        
        return text 
    
    
    def extract_doc(self):
        """doc
        * **Extract each paper at a time**
        * NOTE: if sended one document then it just return document dicctonary else it will
                return list of document 
        """
        
        ext = os.path.splitext(self.doc_url)[1].lower()
        
        # pdf_lamma_reader = pymupdf4llm.to_markdown()
        # json_lamma_reader = pymupdf4llm.to_json()
        if ext == '.pdf':
            pdf_text = self.extract_pdf(self.doc_url)
            return pdf_text
        elif ext == '.html' or ext == '.htm':
            html_text = self.extract_html(self.doc_url)
            return html_text
        elif ext == '.md':
            md_text = self.extract_markdown(self.doc_url)
            return md_text
        else:
            raise ValueError(f"Unsupported file type: {ext}")


