---
language: javascript
type: error_fix
tags: reference, undefined
---
# ReferenceError: is not defined
## Problem
Variable used before declaration.
## Fix
1. Declare with let/const: const x = 1
2. Check scope - move declaration up
3. Import module: import { x } from './module.js'
