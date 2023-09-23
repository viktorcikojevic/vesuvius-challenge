from transformers import TrainerCallback
import numpy as np

class SaveBestThresholdCallback(TrainerCallback):
    "A callback to save the best threshold"
    def __init__(self):
        self.best_threshold = 0.0
        self.best_fbeta = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'eval_fbeta' in logs:
            if logs['eval_fbeta'] > self.best_fbeta:
                self.best_fbeta = logs['eval_fbeta']
                self.best_threshold = logs.get('eval_threshold', 0.0)  # Make sure to also log 'threshold' in your compute_fbeta_score function
