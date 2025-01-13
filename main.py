#!/usr/bin/env python3
import argparse
import json
from typing import List, Dict, Any

import datasets
from openai import OpenAI


def create_prompt(query: str, tools: str) -> str:
    """Create a prompt for the model that includes the query and available tools."""
    tools_dict = json.loads(tools)
    tools_description = json.dumps(tools_dict, indent=2)

    return f"""Given the following user query and available tools, generate appropriate tool calls in JSON format.

User Query: {query}

Available Tools:
{tools_description}

Generate tool calls in this exact format:
[{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}]

Your response should ONLY include the JSON array of tool calls, nothing else.
"""


def validate_tool_call(tool_call: Dict[str, Any], available_tools: List[Dict[str, Any]]) -> bool:
    """Validate that a tool call matches the available tools schema."""
    if not isinstance(tool_call, dict) or "name" not in tool_call or "arguments" not in tool_call:
        return False

    # Find matching tool definition
    tool_def = next(
        (t for t in available_tools if t["name"] == tool_call["name"]), None)
    if not tool_def:
        return False

    # Validate arguments against parameters
    if "parameters" in tool_def:
        required_params = {
            k: v for k, v in tool_def["parameters"].items()
            if isinstance(v, dict) and v.get("default") is None
        }
        for param in required_params:
            if param not in tool_call["arguments"]:
                return False

    return True


def process_dataset(
    ds: datasets.Dataset, model: str, base_url: str, api_key: str
) -> datasets.Dataset:
    """Process the dataset and generate tool calls using the LLM."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    my_answers = []

    for query, tools in zip(ds["query"], ds["tools"]):
        try:
            # Create prompt with query and tools
            prompt = create_prompt(query, tools)

            # Call the model
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful AI that generates tool calls based on user queries."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.1  # Lower temperature for more consistent outputs
            )

            # Extract and validate the response
            answer = response.choices[0].message.content.strip()
            try:
                # Parse the response as JSON
                tool_calls = json.loads(answer)

                # Validate tool calls
                tools_dict = json.loads(tools)
                if isinstance(tool_calls, list) and all(
                    validate_tool_call(call, tools_dict if isinstance(
                        tools_dict, list) else [tools_dict])
                    for call in tool_calls
                ):
                    formatted_answer = json.dumps(tool_calls)
                else:
                    formatted_answer = "[]"  # Empty tool calls if validation fails
            except json.JSONDecodeError:
                formatted_answer = "[]"

        except Exception as e:
            print(f"Error processing query: {query}")
            print(f"Error details: {str(e)}")
            formatted_answer = "[]"

        my_answers.append(formatted_answer)

    return ds.add_column("my_answers", my_answers)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tool calls using an LLM")
    parser.add_argument("--model", required=True,
                        help="Name of the model to use")
    parser.add_argument("--base_url", required=True,
                        help="Base URL of the inference server")
    parser.add_argument("--api_key", required=True,
                        help="API key for the inference server")
    parser.add_argument("--dev", action="store_true",
                        help="Run on first 100 examples only")
    parser.add_argument("--train", action="store_true",
                        help="Process training data without batches")
    args = parser.parse_args()

    # Load the dataset from local directory
    print("Loading dataset...")
    ds = datasets.load_dataset('parquet',
                               data_files='./dataset/data/train-00000-of-00001.parquet')['train']
    print(f"Dataset loaded with {len(ds)} examples")

    if args.train:
        print("Training mode: Processing all examples without batches")
        processed_batch = process_dataset(
            ds, args.model, args.base_url, args.api_key)
        all_results = processed_batch['my_answers']
    elif args.dev:
        ds = ds.select(range(100))
        print(f"Dev mode: Processing first 100 examples with batch size 10")
        batch_size = 10
        all_results = []
        for i in range(0, len(ds), batch_size):
            end_idx = min(i + batch_size, len(ds))
            print(f"Processing batch {i} to {end_idx} of {len(ds)}")
            batch = ds.select(range(i, end_idx))
            processed_batch = process_dataset(
                batch, args.model, args.base_url, args.api_key)
            all_results.extend(processed_batch['my_answers'])
    else:
        print(
            f"Full mode: Processing all {len(ds)} examples with batch size 1000")
        batch_size = 1000
        all_results = []
        for i in range(0, len(ds), batch_size):
            end_idx = min(i + batch_size, len(ds))
            print(f"Processing batch {i} to {end_idx} of {len(ds)}")
            batch = ds.select(range(i, end_idx))
            processed_batch = process_dataset(
                batch, args.model, args.base_url, args.api_key)
            all_results.extend(processed_batch['my_answers'])

    # Create final dataset with all results
    final_ds = ds.add_column("my_answers", all_results)

    # Save the resulting dataset
    print("Saving results...")
    final_ds.save_to_disk("./my_dataset")
    print("Done!")


if __name__ == "__main__":
    main()
