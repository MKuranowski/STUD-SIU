from pathlib import Path

from src.turtle_estimator import load_models_csv, save_models_csv

filename = Path("dqn_multi_models.csv")
save_models_csv(filename, load_models_csv(filename))
print(f"Dane zostały posortowane i zapisane w pliku '{filename}'.")
