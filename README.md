# Verified Findings — Plearun Take-Home

Offline Python 3.10+, standard library only; **no network access or LLM required**.

```bash
python3 plearun_cli.py reports/report_fictional_02.txt records/ > findings.json
make test
```

**How it works.** `claims.py` extracts verbatim report propositions. `evidence.py` reads UTF-8-BOM/marker CSV, Windows-1252 notes, nested JSON (ISO, local and Unix time), and overlapping WebVTT cues. It retains exact raw text and physical line numbers (or VTT cue numbers). `verifier.py` links claims to specific events, prioritizing defensible findings. Every `match`/`conflict` checks that its quoted text is a literal substring of the cited source row; otherwise it becomes `not_settled`. Records are treated only as data, never instructions. `plearun_cli.py` produces the specified JSON.

**Three hardest decisions.** (1) **Exact provenance:** source row/quote, rather than reconstructed CSV or JSON values. (2) **Unsynchronized clocks:** estimate an offset only from at least two different identifiable events with consistent differences; anonymous camera stops are not anchors. (3) **Avoid false conflicts:** a matching camera number does not identify a vehicle; without an explicit vehicle link, abstain. Spoken evidence supports what an officer reported, not independent speed calibration.

**What it gets wrong.** Regex claim extraction misses complicated paraphrases, compound or implicit references, and non-English text. It cannot always connect different sources to the same car, and single-row citations cannot express the full context of a multi-event calculation. Some genuine claims remain unresolved.

**What would break first on real records.** Scanned PDFs, changing device-clock drift, large JSON exports (100 MB/file guard), ambiguous identities, and unusually structured records require stronger parsers, entity linking and alignment.

**AI/tool disclosure.** ChatGPT (OpenAI GPT-6) assisted with architecture, implementation, regression tests and review. The tool itself runs using Python's standard library; `unittest`/`make test` run offline. No runtime LLM is used. 
