# Expected tiny smoke test

Build:

```bash
python train_memory_hf.py --csv ./example.csv --output-dir ./local-model
```

Start the service:

```bash
MODEL_DIR=./local-model API_KEY=local-eval-key python serve_openai.py
```

Question-only query `What is 2 + 2?` should return `4`.

Choice query:

```text
What is 2 + 2?
A) 5
B) 6
C) 4
D) 3
```

should return `ANSWER: C`.
