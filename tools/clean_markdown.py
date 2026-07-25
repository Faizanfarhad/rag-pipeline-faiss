import re

def clean_markdown(text):
    """
    Remove Markdown/HTML tags while preserving plain text content
    """
    # Remove bold/italic: **text** or *text*
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove sup/sub HTML tags: <sup>text</sup>
    text = re.sub(r'<sup>(.*?)</sup>', r'\1', text)
    text = re.sub(r'<sub>(.*?)</sub>', r'\1', text)
    
    # Remove links: [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Remove blockquotes: > text -> text
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Remove headers: # text -> text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
