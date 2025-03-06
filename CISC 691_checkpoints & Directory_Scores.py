import os
import json

# List of checkpoint directories
checkpoint_dirs = [
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-63",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-126",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-189",
     r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-252",
      r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-315",
       r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-378",
       r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\fine_tuned_model"
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



import os
import json

# List of checkpoint directories (including the final fine_tuned_model directory)
checkpoint_dirs = [
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-63",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-126",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-189",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-252",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-315",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\checkpoint-378",
    r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A03\fine_tuned_model",
]

for cp_dir in checkpoint_dirs:
    print(f"\nMetrics for directory: {cp_dir}")
    trainer_state_path = os.path.join(cp_dir, "trainer_state.json")
    
    # Try loading trainer_state.json first
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Iterate through log_history entries to extract evaluation metrics
        found = False
        for entry in state.get("log_history", []):
            if "eval_accuracy" in entry and "eval_f1" in entry:
                step = entry.get("step", "N/A")
                accuracy = entry.get("eval_accuracy")
                f1 = entry.get("eval_f1")
                print(f"  Step: {step} | Accuracy: {accuracy} | F1: {f1}")
                found = True
        if not found:
            print("  No evaluation metrics found in trainer_state.json.")
    else:
        print(f"trainer_state.json not found in {cp_dir}.")
        # Try loading eval_results.json if available
        eval_results_path = os.path.join(cp_dir, "eval_results.json")
        if os.path.exists(eval_results_path):
            with open(eval_results_path, "r", encoding="utf-8") as f:
                eval_results = json.load(f)
            print("Evaluation results:")
            for key, value in eval_results.items():
                print(f"  {key}: {value}")
        else:
            print("  No evaluation results file found in this directory.")
