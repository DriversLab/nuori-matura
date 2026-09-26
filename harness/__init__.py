"""Exam harness for the CKE-style history matura ("separate-text-and-images-v1" packages).

Modules:
  exam_io      load exam.json / answers-template.json, write answers.json, validate it like the upload site
  prompts      organizer system prompt, item kinds, answer-format specs, chat messages (train and serve)
  client       OpenAI-compatible chat client (llama-server / QVAC) with retries and thread-pool concurrency
  postprocess  turn raw model output into the final answer string (closed-syntax normalisation, essay checks)
  vision       (builder B) image descriptions pre-pass

Entry points: scripts/run_exam.py, scripts/validate_answers.py.
Nothing here loads a model; the only network access is to the --base-url you pass to the client.
"""
