import os
import json

# List of checkpoint directories
checkpoint_dirs = [
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-63",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-126",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-189",
]

# Iterate over each checkpoint directory and read trainer_state.json
for cp_dir in checkpoint_dirs:
    trainer_state_path = os.path.join(cp_dir, "trainer_state.json")
    try:
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except FileNotFoundError:
        print(f"trainer_state.json not found in {cp_dir}")
        continue

    print(f"\nMetrics for checkpoint directory: {cp_dir}")
    for entry in state.get("log_history", []):
        if "eval_accuracy" in entry and "eval_f1" in entry:
            step = entry.get("step", "N/A")
            accuracy = entry.get("eval_accuracy")
            f1 = entry.get("eval_f1")
            print(f"  Step: {step} | Accuracy: {accuracy} | F1: {f1}")
