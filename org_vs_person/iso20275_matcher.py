"""ISO 20275 legal form suffix matcher.

This module provides deterministic extraction of legal form suffixes from
organization names using the ISO 20275 Entity Legal Form (ELF) standard.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from org_vs_person.normalization import normalize, NormalizedText


# Ambiguous short forms that frequently collide with personal names (Tier C)
AMBIGUOUS_SHORT_FORMS = {
    "sa", "ag", "ab", "nv", "bv", "oy", "as", "os", "se", "sp",
    "sae", "sas", "srl", "sro", "spa", "kft", "kk", "pte", "pty"
}


@dataclass
class FormMetadata:
    """Metadata for a legal form from ISO 20275.
    
    Attributes:
        elf_code: Entity Legal Form code
        country: Country of formation (ISO 3166-1)
        language: Language code (ISO 639-1)
        canonical_label: Original legal form name
        tier: Confidence tier (A, B, or C)
    """
    elf_code: str
    country: str
    language: str
    canonical_label: str
    tier: str


@dataclass
class SuffixMatch:
    """Result of matching a legal form suffix.
    
    Attributes:
        suffix: The matched suffix string (normalized)
        metadata: FormMetadata for the matched legal form
        token_len: Number of tokens in the suffix
        char_len: Character length of suffix (excluding spaces)
        is_ambiguous_short: Whether this is a Tier C ambiguous form
    """
    suffix: str
    metadata: FormMetadata
    token_len: int
    char_len: int
    is_ambiguous_short: bool


class ISO20275Matcher:
    """Matcher for ISO 20275 legal form suffixes.
    
    Provides fast, deterministic suffix matching using longest-match-first
    algorithm with confidence tier classification.
    """
    
    def __init__(self, iso_csv_path: Optional[Path] = None):
        """Initialize the matcher and load ISO 20275 data.
        
        Args:
            iso_csv_path: Path to ISO 20275 CSV file. If None, uses default.
        """
        if iso_csv_path is None:
            # Default path relative to package
            package_dir = Path(__file__).parent.parent
            iso_csv_path = package_dir / "data" / "iso20275" / "2023-09-28-elf-code-list-v1.5.csv"
        
        self.iso_csv_path = iso_csv_path
        self.forms_by_len: Dict[int, Dict[str, FormMetadata]] = {}
        self._load_forms()
    
    def _load_forms(self):
        """Load and preprocess ISO 20275 legal forms from CSV."""
        if not self.iso_csv_path.exists():
            raise FileNotFoundError(
                f"ISO 20275 CSV not found at {self.iso_csv_path}. "
                "Please ensure the data file is available."
            )
        
        # Use utf-8-sig to handle BOM (Byte Order Mark) in CSV
        with open(self.iso_csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                elf_code = row.get('ELF Code', '').strip()
                country = row.get('Country of formation', '').strip()
                language = row.get('Language', '').strip()
                
                # Get legal form name and abbreviations
                legal_form_name = row.get('Entity Legal Form name Local name', '').strip()
                abbreviations_local = row.get('Abbreviations Local language', '').strip()
                
                # Skip rows with no ELF code or inactive status
                status = row.get('ELF Status ACTV/INAC', '').strip()
                if not elf_code or status == 'INAC':
                    continue
                
                # Process legal form name
                if legal_form_name:
                    self._add_form(legal_form_name, elf_code, country, language)
                
                # Process abbreviations (semicolon-separated)
                if abbreviations_local:
                    for abbrev in abbreviations_local.split(';'):
                        abbrev = abbrev.strip()
                        if abbrev:
                            self._add_form(abbrev, elf_code, country, language)
    
    def _add_form(self, form_text: str, elf_code: str, country: str, language: str):
        """Add a legal form to the lookup structure.
        
        Args:
            form_text: Legal form text (name or abbreviation)
            elf_code: ELF code
            country: Country code
            language: Language code
        """
        normalized = normalize(form_text, strip_diacritics_flag=False)
        if not normalized.tokens:
            return
        
        # Determine tier based on form characteristics
        tier = self._determine_tier(normalized)
        
        # Create metadata
        metadata = FormMetadata(
            elf_code=elf_code,
            country=country,
            language=language,
            canonical_label=form_text,
            tier=tier
        )
        
        # Store by token length
        token_len = len(normalized.tokens)
        suffix_key = normalized.norm_str
        
        if token_len not in self.forms_by_len:
            self.forms_by_len[token_len] = {}
        
        # Only store if not already present (first wins)
        if suffix_key not in self.forms_by_len[token_len]:
            self.forms_by_len[token_len][suffix_key] = metadata
    
    def _determine_tier(self, normalized: NormalizedText) -> str:
        """Determine confidence tier for a legal form.
        
        Args:
            normalized: Normalized legal form text
            
        Returns:
            'A', 'B', or 'C' tier classification
        """
        token_count = len(normalized.tokens)
        char_len = len(normalized.norm_str.replace(' ', ''))
        
        # Check if it's in the ambiguous short forms set
        if normalized.norm_str in AMBIGUOUS_SHORT_FORMS:
            return 'C'
        
        # Tier A: Multi-token forms OR single-token >= 4 chars
        if token_count >= 2 or char_len >= 4:
            return 'A'
        
        # Tier B: Short forms (2-3 chars) not in ambiguous list
        if char_len in [2, 3]:
            return 'B'
        
        # Edge case: single char (very rare, conservative approach)
        return 'C'
    
    def match_legal_form(self, text: str) -> Optional[SuffixMatch]:
        """Match a legal form suffix in the given text.
        
        Uses longest-match-first algorithm on last k tokens where k=min(6, token_count).
        
        Args:
            text: Input text (name) to match
            
        Returns:
            SuffixMatch if a legal form is found, None otherwise
        """
        normalized = normalize(text, strip_diacritics_flag=False)
        if not normalized.tokens:
            return None
        
        # Consider last k tokens
        k = min(6, len(normalized.tokens))
        
        # Longest-match-first: try k tokens down to 1
        for L in range(k, 0, -1):
            # Get last L tokens
            suffix_tokens = normalized.tokens[-L:]
            suffix_str = ' '.join(suffix_tokens)
            
            # Check if this suffix exists in our lookup
            if L in self.forms_by_len and suffix_str in self.forms_by_len[L]:
                metadata = self.forms_by_len[L][suffix_str]
                
                # Calculate character length (excluding spaces)
                char_len = len(suffix_str.replace(' ', ''))
                
                return SuffixMatch(
                    suffix=suffix_str,
                    metadata=metadata,
                    token_len=L,
                    char_len=char_len,
                    is_ambiguous_short=(metadata.tier == 'C')
                )
        
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded forms.
        
        Returns:
            Dictionary with counts by tier and total
        """
        total = 0
        tier_counts = {'A': 0, 'B': 0, 'C': 0}
        
        for forms_dict in self.forms_by_len.values():
            for metadata in forms_dict.values():
                total += 1
                tier_counts[metadata.tier] += 1
        
        return {
            'total': total,
            'tier_a': tier_counts['A'],
            'tier_b': tier_counts['B'],
            'tier_c': tier_counts['C']
        }
