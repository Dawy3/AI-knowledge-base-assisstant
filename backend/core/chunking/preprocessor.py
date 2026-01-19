"""
Text Preprocessing for RAG Pipeline.

FOCUS: Clean text, remove noise
MUST: Normalize whitespace, handle special chars
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    
    # Whitespace normalization
    normalize_whitespace: bool = True
    collapse_newlines: bool = True
    max_consecutive_newlines: int = 2
    
    # Unicode handling
    normalize_unicode: bool = True
    unicode_form: str = "NFKC"  # NFC, NFKC, NFD, NFKD
    remove_control_chars: bool = True
    
    # Character handling
    remove_zero_width_chars: bool = True
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    
    # Content cleaning
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    
    # Header/footer removal (for PDFs)
    remove_headers_footers: bool = False
    header_footer_max_lines: int = 3
    
    # Special content
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    
    # Length constraints
    min_chunk_chars: int = 50
    max_chunk_chars: int = 50000
    
    # Custom patterns to remove
    custom_patterns: list[str] = field(default_factory=list)


class TextPreprocessor:
    """
    Text preprocessor for cleaning and normalizing text before chunking.
    
    FOCUS: Clean text, remove noise
    MUST: Normalize whitespace, handle special chars
    
    Processing pipeline:
    1. Unicode normalization
    2. Control character removal
    3. Whitespace normalization
    4. Special character normalization
    5. Content cleaning (URLs, emails, etc.)
    6. Custom pattern removal
    """
    
    # Zero-width characters to remove
    ZERO_WIDTH_CHARS = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # BOM
        '\u00ad',  # Soft hyphen
    ]
    
    # Quote normalization mapping
    QUOTE_MAP = {
        '"': '"', '"': '"',  # Double quotes
        ''': "'", ''': "'",  # Single quotes
        '‹': "'", '›': "'",  # Single angle quotes
        '«': '"', '»': '"',  # Double angle quotes
        '„': '"', '‟': '"',  # Low quotes
    }
    
    # Dash normalization mapping
    DASH_MAP = {
        '–': '-',  # En dash
        '—': '-',  # Em dash
        '―': '-',  # Horizontal bar
        '‐': '-',  # Hyphen
        '‑': '-',  # Non-breaking hyphen
        '⁃': '-',  # Hyphen bullet
    }
    
    # Regex patterns
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*',
        re.IGNORECASE
    )
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    PHONE_PATTERN = re.compile(
        r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
    )
    WHITESPACE_PATTERN = re.compile(r'[ \t]+')
    NEWLINE_PATTERN = re.compile(r'\n{3,}')
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or PreprocessingConfig()
        self._custom_patterns = [
            re.compile(p) for p in self.config.custom_patterns
        ]
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not text.strip():
            return ""
        
        # Step 1: Unicode normalization
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Step 2: Remove control characters
        if self.config.remove_control_chars:
            text = self._remove_control_chars(text)
        
        # Step 3: Remove zero-width characters
        if self.config.remove_zero_width_chars:
            text = self._remove_zero_width_chars(text)
        
        # Step 4: Normalize quotes and dashes
        if self.config.normalize_quotes:
            text = self._normalize_quotes(text)
        if self.config.normalize_dashes:
            text = self._normalize_dashes(text)
        
        # Step 5: Content cleaning
        if self.config.remove_urls:
            text = self._remove_urls(text)
        if self.config.remove_emails:
            text = self._remove_emails(text)
        if self.config.remove_phone_numbers:
            text = self._remove_phone_numbers(text)
        
        # Step 6: Whitespace normalization (do this after content removal)
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)
        if self.config.collapse_newlines:
            text = self._collapse_newlines(text)
        
        # Step 7: Custom pattern removal
        for pattern in self._custom_patterns:
            text = pattern.sub('', text)
        
        # Step 8: Final trim
        text = text.strip()
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to standard form."""
        return unicodedata.normalize(self.config.unicode_form, text)
    
    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except newlines and tabs."""
        return ''.join(
            char for char in text
            if not unicodedata.category(char).startswith('C')
            or char in '\n\t\r'
        )
    
    def _remove_zero_width_chars(self, text: str) -> str:
        """Remove zero-width and invisible characters."""
        for char in self.ZERO_WIDTH_CHARS:
            text = text.replace(char, '')
        return text
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize various quote styles to standard ASCII."""
        for fancy, standard in self.QUOTE_MAP.items():
            text = text.replace(fancy, standard)
        return text
    
    def _normalize_dashes(self, text: str) -> str:
        """Normalize various dash styles to standard hyphen."""
        for fancy, standard in self.DASH_MAP.items():
            text = text.replace(fancy, standard)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize horizontal whitespace (spaces and tabs)."""
        # Replace multiple spaces/tabs with single space
        text = self.WHITESPACE_PATTERN.sub(' ', text)
        # Clean up spaces around newlines
        text = re.sub(r' *\n *', '\n', text)
        return text
    
    def _collapse_newlines(self, text: str) -> str:
        """Collapse excessive consecutive newlines."""
        max_newlines = self.config.max_consecutive_newlines
        replacement = '\n' * max_newlines
        return self.NEWLINE_PATTERN.sub(replacement, text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.URL_PATTERN.sub(' ', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.EMAIL_PATTERN.sub(' ', text)
    
    def _remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text."""
        return self.PHONE_PATTERN.sub(' ', text)
    
    def clean_for_embedding(self, text: str) -> str:
        """
        Light cleaning optimized for embedding generation.
        
        Preserves more structure than full preprocessing.
        """
        if not text:
            return ""
        
        # Basic normalization only
        text = unicodedata.normalize('NFKC', text)
        text = self._remove_zero_width_chars(text)
        text = self._normalize_whitespace(text)
        text = text.strip()
        
        return text
    
    def extract_metadata_text(self, text: str) -> dict[str, str]:
        """
        Extract and separate metadata from main content.
        
        Returns dict with 'content' and optional metadata fields.
        """
        result = {'content': text}
        
        # Extract URLs before removing
        urls = self.URL_PATTERN.findall(text)
        if urls:
            result['urls'] = urls
        
        # Extract emails before removing
        emails = self.EMAIL_PATTERN.findall(text)
        if emails:
            result['emails'] = emails
        
        return result
    
    def remove_boilerplate(
        self,
        text: str,
        header_pattern: Optional[str] = None,
        footer_pattern: Optional[str] = None
    ) -> str:
        """
        Remove headers, footers, and other boilerplate content.
        
        Useful for PDFs with repeated page headers/footers.
        """
        lines = text.split('\n')
        
        if self.config.remove_headers_footers:
            max_lines = self.config.header_footer_max_lines
            
            # Simple heuristic: remove short repeated lines at start/end
            # More sophisticated detection would use pattern matching
            if len(lines) > max_lines * 2:
                # Check for header pattern
                if header_pattern:
                    header_re = re.compile(header_pattern)
                    lines = [l for l in lines if not header_re.match(l)]
                
                # Check for footer pattern
                if footer_pattern:
                    footer_re = re.compile(footer_pattern)
                    lines = [l for l in lines if not footer_re.match(l)]
        
        return '\n'.join(lines)


def create_preprocessor(
    normalize_whitespace: bool = True,
    remove_urls: bool = False,
    **kwargs
) -> TextPreprocessor:
    """
    Factory function to create a preprocessor with custom settings.
    
    Args:
        normalize_whitespace: Whether to normalize whitespace
        remove_urls: Whether to remove URLs
        **kwargs: Additional PreprocessingConfig parameters
        
    Returns:
        Configured TextPreprocessor instance
    """
    config = PreprocessingConfig(
        normalize_whitespace=normalize_whitespace,
        remove_urls=remove_urls,
        **kwargs
    )
    return TextPreprocessor(config)