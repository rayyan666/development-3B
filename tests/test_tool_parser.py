from app.agent.tool_parser import ToolParser

sample_output = """
{
  "tool_call": {
    "name": "train_model",
    "parameters": {
      "model_type": "RandomForest",
      "dataset_id": "default",
      "target_column": "DailyHours"
    }
  }
}
Extra commentary here.
"""

parsed = ToolParser.parse(sample_output)

print(parsed)
