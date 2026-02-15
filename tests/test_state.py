from app.state.dataset_registry import DatasetRegistry
import pandas as pd

df = pd.DataFrame({"a": [1,2,3]})
DatasetRegistry.register("test", df)

print(DatasetRegistry.list_datasets())
print(DatasetRegistry.get("test"))
