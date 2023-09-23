import torch
from transformers import Trainer, TrainerCallback
import numpy as np
from tqdm import tqdm
import wandb

from .metrics import fbeta_score


class VesuviusDataCollator():
    def __call__(self, batch):
        frag_crop = torch.stack([item[0] for item in batch])
        mask_crop = torch.stack([item[1] for item in batch])
        
        return {"input_ids": frag_crop, "labels": mask_crop}

class SaveBestThresholdCallback(TrainerCallback):
    def __init__(self, save_path, metric_name="eval_fbeta"):
        self.best_metric = float('-inf')
        self.save_path = save_path
        self.metric_name = metric_name

    def on_evaluate(self, args, state, control, **kwargs):
        logs = kwargs['logs']
        current_metric = logs.get(self.metric_name, None)
        
        if current_metric is None:
            print(f"Metric {self.metric_name} not found. Make sure it is returned by the evaluate function.")
            return

        if current_metric > self.best_metric:
            print(f"New best {self.metric_name} of {current_metric}. Saving model to {self.save_path}")
            self.best_metric = current_metric

            # Save the model
            model = kwargs['model']
            torch.save(model.state_dict(), self.save_path)
            
class VesuviusTrainer(Trainer):
    def __init__(self, criterion, eval_on_n_batches, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion
        self.eval_on_n_batches = eval_on_n_batches

    def compute_loss(self, model, inputs, return_outputs=False):
        mask_crop = inputs.pop("labels")
        output_logits = model(**inputs)
        loss = self.criterion(output_logits, mask_crop)
        return (loss, output_logits) if return_outputs else loss
    
    
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        thresholds = np.linspace(0, 1, 100)
        fbeta_scores_list = []

        for batch_indx, batch in tqdm(enumerate(eval_dataloader), total=self.eval_on_n_batches):
            if batch_indx >= self.eval_on_n_batches:
                break
            
            batch = self._prepare_inputs(batch)
            labels = batch['labels']
            with torch.no_grad():
                preds = self.model(**batch)
                preds_sigmoid = torch.sigmoid(preds)
                
                fbeta_scores = []
                for threshold in thresholds:
                    fbeta = fbeta_score(preds_sigmoid, labels, apply_sigmoid=False, threshold=threshold)
                    fbeta = fbeta.detach().cpu().numpy()
                    fbeta_scores.append(fbeta)

                fbeta_scores_list.append(fbeta_scores)

        # Average fbeta across all batches and thresholds
        avg_fbeta_scores = np.mean(np.array(fbeta_scores_list), axis=0).tolist()
        best_fbeta = float(np.max(avg_fbeta_scores))
        best_threshold = float(thresholds[np.argmax(avg_fbeta_scores)])

        metrics = {
            'eval_fbeta': best_fbeta,
            'eval_threshold': best_threshold
        }
        print(metrics)
        
        if self.args.report_to == "wandb":
            
            wandb.log(metrics) 
        
        return metrics
    