"""V11 clean camera pipeline.

No legacy V8/V9/V10 runtime inheritance is allowed in this package.  Each step
adds exactly one layer after the previous step has passed its own acceptance
check.
"""
