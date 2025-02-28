# RecSeqFinder
# Restriction Enzyme Sequence Finder

This Python script analyzes SnapGene `.dna` files to count occurrences of restriction enzyme recognition sequences and generates an Excel output with detailed results. It supports both raw sequence inputs (e.g., `GATC`) and restriction enzyme names (e.g., `EcoRI`), providing flexibility and robust error handling.

## Features

1. **Flexible Input Handling**:
   - Accepts comma-separated recognition sequences (e.g., `GATC, EcoRI`).
   - Converts restriction enzyme names to their corresponding recognition sequences (e.g., `EcoRI` → `GAATTC`) using a predefined dictionary.
   - Case-insensitive enzyme name recognition (e.g., `EcoRI`, `ECORI` both work).

2. **Comprehensive Sequence Analysis**:
   - Counts occurrences of recognition sequences in both forward and reverse-complement strands of DNA sequences.
   - Identifies palindromic sequences and subtracts them from the total count for accurate restriction site analysis.
   - Supports IUPAC ambiguous DNA codes (e.g., `N` for any base, `M` for `A` or `C`).

3. **SnapGene File Parsing**:
   - Reads `.dna` files recursively from a specified directory using the `snapgene_reader` library.
   - Skips files exceeding 100,000 bp to avoid memory issues.

4. **Output Generation**:
   - Produces an Excel file (`result.xlsx`) with four sheets:
     - `All count`: Total non-palindromic restriction site counts (Forward + Reverse - Palindromes).
     - `Forward Read`: Counts on the forward strand.
     - `Reverse Read`: Counts on the reverse-complement strand.
     - `Palindrome Count`: Counts of palindromic occurrences.
   - Sorts results by recognition sequence counts in descending order, then by file name alphabetically.

5. **Error Handling and Debugging**:
   - Robust exception handling with detailed error messages and stack traces for failed file parsing.
   - Extensive logging during sequence processing and data frame creation for troubleshooting.

6. **Customization**:
   - Easily extensible `default_restriction_enzymes` dictionary for adding new restriction enzymes.
   - The current data includes enzymes from the REBASE gold_standard sequences as defaults.
   - Modular design allows adaptation for other sequence analysis tasks.

## Usage

1. **Requirements**:
   - Python 3.x
   - Libraries: `pandas`, `biopython`, `snapgene_reader`, `xlsxwriter`

   Install dependencies:
   ```bash
   pip install pandas biopython snapgene_reader xlsxwriter
