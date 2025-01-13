# ellamind Coding Challenge

> I've provided my solution detail in solution.ipynb file.

Create a Python CLI that uses an LLM to generate tool calling structures based on user queries, matching the format in the provided dataset.

## Requirements

1. [Download](https://huggingface.co/crusoeai/Arcee-Agent-GGUF/blob/main/arcee-agent.Q4_K_M.gguf) the required model: A `Q4_K_M` quantization of [Arcee-Agent](https://huggingface.co/crusoeai/Arcee-Agent-GGUF).

2. Set up an OpenAI-compatible inference server with the model (e.g., using llama-cpp-python, LMStudio, or Ollama). This inference server runs the model and provides an API endpoint, that your code will call.

3. Implement the solution:

   - Load the dev dataset from `dataset/`
   - _Note: the "answers" column won't be available in the private test set when grading, but you can use them for development/evaluation._
   - Process queries to generate tool calls matching the dataset format
   - Add results in a new column `"my_answers"`
   - Save results to `my_dataset/`

4. CLI must support these arguments, that are used to call the OpenAI-compatible inference server that runs the model:
   - `--model`: model name
   - `--base_url`: server URL (format: `http://<domain>:<port>/v1`)
   - `--api_key`: API key for the server

_See [main.py](main.py) for the expected CLI usage and a skeleton implementation._

## Setup & Execution

1. Install dependencies:

```bash
poetry install --no-root
```

2. Download the model, e.g. with:

```bash
poetry run huggingface-cli download crusoeai/Arcee-Agent-GGUF arcee-agent.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

3. Spin up an OpenAI-compatible inference server running the model.

4. Your solution must be callable in the following way (example args):

```bash
poetry run python main.py --model crusoeai/Arcee-Agent-GGUF --base_url http://localhost:8000/v1 --api_key abc123
```

## Submission

- Create a private GitHub repository
- Invite @ellamind-admin
- Ensure large files are gitignored

## Additional Notes

- The dev dataset is available in `dataset/` with example usage in `dataset.ipynb`
- Grading will use a private test-set with the same structure, without the "answers" column.
- Note that you don't need to use "OpenAI Tool Calling" format and can just use the raw LLM output and parse it as appropriate.
- If you have hardware/OS constraints, contact us
- Time investment: couple of hours max
- We would expect the result within a week of receiving the challenge
- Perfect results aren't expected
