---
language: python
type: error_fix
tags: syntax, IndentationError
---
# IndentationError
## Problem
Unexpected indent or unindent.
## Causes
- Mixing tabs and spaces
- Wrong indentation level
## Fix
1. Set editor to 4 spaces, no tabs
2. Run: python -m tabnanny yourfile.py
3. Auto-fix: autopep8 --in-place --aggressive file.py
