---
description: General debugging and problem-solving guidelines
---

When debugging code, addressing errors, or implementing fixes, ALWAYS observe the following general guidelines:

1. **Think Deeper than the Error Message**: Do not blindly react to the exact phrasing of an error trace or warning (e.g. string matching "memory" or immediately cloning a tensor to bypass a warning).
2. **Understand the Root Cause**: Investigate *why* the error occurred or the warning was generated within the broader architectural flow of the system.
3. **Solve the Real Problem**: Implement fixes that address the root cause elegantly, rather than creating naive patches or workarounds that simply satisfy the literal error string but add unnecessary overhead or complexity.
