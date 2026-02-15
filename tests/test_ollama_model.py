import requests
import time
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "ds-ml-assistant"


def query_ollama(prompt: str):
    start_time = time.time()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    end_time = time.time()

    duration = end_time - start_time

    if response.status_code != 200:
        return {
            "error": response.text,
            "time_taken": duration
        }

    output = response.json()["response"]

    return {
        "response": output.strip(),
        "time_taken": duration
    }


def is_valid_tool_call(output: str):
    try:
        parsed = json.loads(output)
        return "tool_call" in parsed
    except Exception:
        return False


def run_tests():

    test_prompts = [
        "Train a random forest model to predict DailyHours.",
        "What is R2 score?",
        "Evaluate the last trained model.",
        "Show summary statistics of the dataset."
    ]

    results = []

    for prompt in test_prompts:
        print("\n===================================")
        print(f"Prompt: {prompt}")

        result = query_ollama(prompt)

        if "error" in result:
            print("Error:", result["error"])
            continue

        output = result["response"]
        duration = result["time_taken"]

        print("\nResponse:")
        print(output)
        print(f"\nTime Taken: {duration:.2f} seconds")

        tool_call_detected = is_valid_tool_call(output)
        print(f"Tool Call JSON Detected: {tool_call_detected}")

        results.append({
            "prompt": prompt,
            "time_taken": duration,
            "tool_call": tool_call_detected
        })

    print("\n===================================")
    print("SUMMARY")
    print("===================================")

    avg_time = sum(r["time_taken"] for r in results) / len(results)

    print(f"Average Response Time: {avg_time:.2f} seconds")
    print(f"Tool Calls Detected: {sum(r['tool_call'] for r in results)} / {len(results)}")


if __name__ == "__main__":
    run_tests()
