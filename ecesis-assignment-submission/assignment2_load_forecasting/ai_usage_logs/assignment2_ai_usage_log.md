# Assignment 2 AI Usage Log

## Tool Used
Codex was used to read the assignment requirements, inspect the provided data, design the forecasting workflow, implement the Python notebook/pipeline, generate outputs, and validate schemas and metrics.

## User Instructions
- Complete Assignment 2 according to `Summer2026-main/README.md`.
- Use local data path `2026 Summer Intern-20260524T050101Z-3-001/2026 Summer Intern`.
- Use Parquet for full forecast files and small CSV files for samples.
- Include all bus rows and treat missing `pd` as zero.
- Use a GPU Transformer as the primary model rather than an MLP.
- Impute historical `pd` only for partial-missing buses that also have nonzero historical load; all-missing, all-zero, and zero-or-missing-only buses stay unfilled and fall back to 0 downstream.

## AI-Assisted Work
- Interpreted the README requirements and suggested approach.
- Verified input parquet schemas, row counts, date ranges, and bus/zone consistency.
- Implemented a CUDA PyTorch Transformer zone model plus bus-share allocation pipeline.
- Added non-leaking 5-level time-aware historical imputation for partial-missing active bus `pd` values used in training, shares, and baselines.
- Added previous-week, previous-year, and bus-hour-weekday historical-average baselines.
- Added leakage-control checks and summary reporting.
- Rebuilt the full forecast output with Transformer model names after confirming the machine has an RTX 3080.

## Human Review Notes
The final forecasts, metrics, and report should be reviewed by the candidate before submission. The submitted chat history can be exported from this Codex session if the recruiting process requires the full transcript.