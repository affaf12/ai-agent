---
language: python
type: error_fix
tags: import, ModuleNotFoundError
---
# ModuleNotFoundError
## Problem
Python cannot find the module you're importing.
## Causes
- Missing __init__.py in package folder
- Wrong sys.path
- Virtual environment not activated
- Typo in module name
## Fix
1. Add empty __init__.py to make folder a package
2. Use absolute import: from mypackage.module import func
3. Check path: import sys; print(sys.path)
4. Activate venv: source venv/bin/activate
