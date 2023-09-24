import torch
from transformers import Trainer, TrainerCallback
import numpy as np
from tqdm import tqdm
import wandb
import os

from .metrics import fbeta_score




class VesuviusDataCollator():
    
    def __init__(self, device):
        self.device = device
    
    def __call__(self, batch):
        
        # send to device
        frag_crop = batch[0].to(self.device)
        mask_crop = batch[1].to(self.device)
        
        return {"frag_crop": frag_crop, "mask_crop": mask_crop}

class VesuviusTrainer():
    
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model")
        self.criterion = kwargs.pop("criterion")
        self.learning_rate = kwargs.pop("learning_rate")
        self.batch_size = kwargs.pop("batch_size")
        self.save_steps = kwargs.pop("save_steps")
        self.total_epochs = kwargs.pop("total_epochs")
        self.logging_steps = kwargs.pop("logging_steps")
        self.report_to = kwargs.pop("report_to")
        self.output_dir = kwargs.pop("output_dir")
        self.dataloader_train = kwargs.pop("dataloader_train") if "dataloader_train" in kwargs else None
        self.dataloader_test = kwargs.pop("dataloader_test") if "dataloader_test" in kwargs else None
        self.dataloader_val = kwargs.pop("dataloader_val") if "dataloader_val" in kwargs else None
        self.eval_on_n_batches = kwargs.pop("eval_on_n_batches")
        self.device=kwargs.pop("device")
        
        # prepare stuff
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # constant LR
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, self.learning_rate)
        self.data_collator = VesuviusDataCollator(self.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        mask_crop = inputs.pop("mask_crop")
        output_logits = model(inputs['frag_crop'])
        loss = self.criterion(output_logits, mask_crop)
        return (loss, output_logits) if return_outputs else loss
    
    
    def evaluate(self):


        thresholds = np.linspace(0, 1, 100)
        fbeta_scores_list = []
        eval_losses = []

        self.model.eval()
        for batch_indx, batch in tqdm(enumerate(self.dataloader_test), total=self.eval_on_n_batches):
            if batch_indx >= self.eval_on_n_batches:
                break
            
            
            batch = self.data_collator(batch)
            frag_crop = batch['frag_crop']
            labels = batch['mask_crop']
            with torch.no_grad():
                preds = self.model(frag_crop)
                preds_sigmoid = torch.sigmoid(preds)
                
                # calculate loss
                loss = self.criterion(preds, labels)
                eval_losses.append(loss.detach().cpu().numpy())
                
                fbeta_scores = []
                for threshold in thresholds:
                    fbeta = fbeta_score(preds_sigmoid, labels, apply_sigmoid=False, threshold=threshold)
                    fbeta = fbeta.detach().cpu().numpy()
                    fbeta_scores.append(fbeta)

                fbeta_scores_list.append(fbeta_scores)
        self.model.train()

        eval_loss = np.mean(np.array(eval_losses))

        # Average fbeta across all batches and thresholds
        avg_fbeta_scores = np.mean(np.array(fbeta_scores_list), axis=0).tolist()
        best_fbeta = float(np.max(avg_fbeta_scores))
        best_threshold = float(thresholds[np.argmax(avg_fbeta_scores)])


        
        metrics = {
            'eval_loss': eval_loss,
            'eval_fbeta': best_fbeta,
            'eval_threshold': best_threshold
        }
        
        return metrics
    
    
    def save_model(self):
        
        # save to self.output_dir
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_{self.step}.pt"))
    
    def log_metrics(self, out_dict):
            
        if 'wandb' in self.report_to:
            wandb.log(out_dict)
    
        df_metrics = pd.DataFrame([out_dict])
        
        # concat to metrics.csv in out_dict, if it exists
        metrics_csv = os.path.join(self.output_dir, "metrics.csv")

        if os.path.exists(metrics_csv):
            df_metrics_old = pd.read_csv(metrics_csv)
            df_metrics = pd.concat([df_metrics_old, df_metrics], axis=0)

        df_metrics.to_csv(metrics_csv, index=False)
    
    
    def train(self):
        
        self.model.train()
        
        
        n_steps = len(self.dataloader_train) * self.total_epochs
        
        self.step = 0
        running_train_loss = 0
        
        for step in tqdm(range(n_steps), total=n_steps):
            
            self.model.train()
            
            batch = next(iter(self.dataloader_train))
            batch = self.data_collator(batch)
            
            self.step = step
            loss = self.compute_loss(self.model, batch)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            running_train_loss += loss.item()
            
            if step % self.logging_steps == 0 and step > 0:
                # Log metrics
                metrics = self.evaluate()
                
                out = {
                    'step': step,
                    'train_loss': running_train_loss / self.logging_steps,
                }
                # combine metrics
                out.update(metrics)
                print(out)
                self.log_metrics(out)
                
                running_train_loss = 0
                
            if step % self.save_steps == 0:
                # Save model checkpoint
                self.save_model()
    