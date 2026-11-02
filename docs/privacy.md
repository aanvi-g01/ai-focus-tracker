# Privacy rules (fixed in Week 1)

The logger records only:

| Recorded | Not recorded |
|---|---|
| foreground process name (e.g. `Code.exe`) | window titles, URLs, file names |
| app-switch events | which keys were pressed (only a count) |
| idle start / resume | screen content, screenshots |
| keyboard / mouse event **counts** per 5 s | clipboard, network traffic |

Enforced in code: whitelisted event names (`storage.py`), no content columns in
the schema, and input callbacks that ignore their arguments (`input_counter.py`).
Tests in `tests/test_logger.py` guard these rules. Raw logs stay on the machine;
only structured summaries may go to the LLM API.
