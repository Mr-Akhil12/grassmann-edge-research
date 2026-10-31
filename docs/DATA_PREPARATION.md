# Next-step natural-language corpus preparation

WikiText-2 raw (Salesforce/wikitext) is pinned to dataset commit
`6231e49f19a707241d6b84d9cff60a3a86b85a85`.

Its three Parquet archives are downloaded only on explicit command,
verified against upstream SHA-256 hashes, and stored under ignored `data/`.
Files are **never included in this Git repository**. Source license:
CC BY-SA 3.0 (check original dataset card; underlying text may have
additional attribution requirements).

```bash
python -m pip install -e ".[data]"
python -m scripts.prepare_wikitext --splits train validation
```

This writes byte-for-byte UTF-8 text datasets plus prepared SHA-256 values
and a provenance manifest. The test split remains untouched until the
experiment protocol is frozen.

**Important:** Reading WikiText-2 as UTF-8 bytes is NOT equivalent to the
paper's ~30,522 WordPiece tokenization. Any reported byte-level bits-per-byte
must NOT be compared numerically with the paper's WordPiece perplexity.
A matched WordPiece/13M-model experiment is still required for a faithful
paper reproduction.

Official dataset: https://huggingface.co/datasets/Salesforce/wikitext
