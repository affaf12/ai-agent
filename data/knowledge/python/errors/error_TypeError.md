---
language: python
type: error_fix
tags: type, TypeError
---
# TypeError: 'NoneType' object is not callable
## Problem
Calling a variable that is None as a function.
## Fix
1. Check function returns value: return result
2. Verify import: from module import func (not module.func = None)
3. Add guard: if func: func()
