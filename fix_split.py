#!/usr/bin/env python3
"""Fix train/test split in DaskPipelineRunner.py to use sklearn's train_test_split with shuffle."""
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "MachineLearning/DaskPipelineRunner.py"

with open(filepath, "r") as f:
    content = f.read()

old_code = """        import dask.dataframe as dd
        data_pd = data.compute()
        train_pd = data_pd.head(num_rows_to_train)
        test_pd = data_pd.tail(num_rows_to_test) if num_rows_to_test > 0 else data_pd.head(0)"""

new_code = """        import dask.dataframe as dd
        from sklearn.model_selection import train_test_split as sklearn_split
        data_pd = data.compute()
        
        # Use sklearn train_test_split with shuffle for proper randomization
        label_col = ml_config.get("label", "isAttacker")
        if label_col in data_pd.columns and data_pd[label_col].nunique() > 1:
            train_pd, test_pd = sklearn_split(
                data_pd, test_size=test_size, random_state=42, 
                shuffle=True, stratify=data_pd[label_col]
            )
        else:
            # Fallback without stratify if label column missing or single class
            train_pd, test_pd = sklearn_split(
                data_pd, test_size=test_size, random_state=42, shuffle=True
            )
        
        self.logger.log(f"Train/test split: shuffle=True, random_state=42, stratified={label_col in data_pd.columns}")"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(filepath, "w") as f:
        f.write(content)
    print("SUCCESS: train/test split fix applied")
else:
    print("ERROR: Could not find exact old code block")
    # Debug: show what's around line 524
    lines = content.split('\n')
    for i in range(519, 530):
        if i < len(lines):
            print(f"{i+1}: {lines[i]}")
    sys.exit(1)
