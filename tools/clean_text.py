import re,unicodedata

def clean_text(text: str) -> str:
    """
    Improved PDF text cleaning.
    Handles:
    - **Unicode normalization**
    - **Preserve Math Equations**
    - **Common PDF ligatures**
    - **Hyphenated line breaks**
    - **Extra newlines**
    - **Extra spaces**
    - **Broken spacing around punctuation**
    """
    if text is None:
        return ""
    # Unicode normalization
    
    text = unicodedata.normalize("NFKC", text)
    # Fix common PDF ligatures
    
    ligature_map = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl"
    }
    
    for bad, good in ligature_map.items():
        text = text.replace(bad, good)
    # Fix hyphenated words broken across lines:
    
    # Example: "trans-\nformer" -> "transformer"
    
    # 1. Extract and hide equations using placeholders
    
    # Matches $$...$$, $...$, \begin{...}...\end{...}, and \[...\]
    
    equation_pattern = r"(\$\$.*?\$\$|\$.*?\$|\\begin\{.*?\}.*?\\end\{.*?\}|\\\[.*?\\\])"
    
    equations = re.findall(equation_pattern, text, flags=re.DOTALL)
    # Replace equations with placeholder tokens e.g., ___EQ_0___
    
    for i, eq in enumerate(equations):
        text = text.replace(eq, f" ___EQ_{i}___ ")
    # 2. Run your original regularization pipeline
    
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text) 
    
    text = re.sub(r"[\n\r\t]+", " ", text) 
    
    text = re.sub(r"\s+", " ", text) 
    
    text = re.sub(r"\s+([.,;:!?%)\]])", r"\1", text) 
    
    text = re.sub(r"([(\[])\s+", r"\1", text) 
    
    text = text.strip()
    
    # 3. Put the original equations back
    for i, eq in enumerate(equations):
        text = text.replace(f"___EQ_{i}___", eq)
    # Clean up any accidental double spaces created around equations
    text = re.sub(r"\s+", " ", text) 
    
    return text 
