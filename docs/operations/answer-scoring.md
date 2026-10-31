# Future JSON answer-scoring contract

Future CPU, CUDA and paired-holdout measurements share
`strict-json-object-finite-typed.v1` in `evaluation/exact_json.py`.
The entire response must decode to one JSON object. Duplicate keys at any depth,
nonfinite decoded numbers (including numeric overflow), code fences and malformed
responses are invalid. Correctness requires exactly the reference keys, values
and recursive Python JSON types; booleans do not equal integers.

Large finite floats and arbitrary-precision integers remain valid. `1e400` has
valid JSON numeric syntax but decodes to a nonfinite float and fails this stricter
experimental contract. Valid format and exact answer correctness are separate.

Generation requests, cached measurement identities and result schemas include
the scorer version. Future comparisons reject missing/mismatched scorer versions.
Old receipts, raw generations and R10 reports are never overwritten or relabeled.
Any separately authorized historical rescore needs a distinct identity/report;
no historical rescore has been performed by this implementation change.
