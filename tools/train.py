from loguru import logger
from datetime import datetime
import argparse
import json
import yaml
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
import os

from kaggle_vesuvius.dataloaders import VesuviusDataset
from kaggle_vesuvius.model_builder import build_model
from kaggle_vesuvius.losses import get_loss_function
from kaggle_vesuvius.custom_transformers import VesuviusDataCollator, VesuviusTrainer

def main(config_path: str,
         work_dir: str,
         data_dir: str):
    
    with open(config_path, "rb") as f:
        config = yaml.load(f, yaml.FullLoader)
    
    # Print the config    
    logger.info(json.dumps(config, indent=4))
    
    
    # Create datasets and  dataloaders
    train_dataset = VesuviusDataset(mode='train',
                                          data_dir=data_dir, 
                                          crop_size=config['crop_size'],
                                          eval_on=config['eval_on'],
                                          z_start=config['z_start'],
                                          z_end=config['z_end'],
                                          stride=config['stride'],
                                          )
    
    val_dataset = VesuviusDataset(mode='val',
                                          data_dir=data_dir, 
                                          crop_size=config['crop_size'],
                                          eval_on=config['eval_on'],
                                          z_start=config['z_start'],
                                          z_end=config['z_end'],
                                          stride=config['stride'],
                                          )
    
    
    dataloader_train = DataLoader(train_dataset, 
                                  batch_size=config['batch_size'],
                                  num_workers=2,
                                  shuffle=True, 
                                  pin_memory=True, 
                                  drop_last=True)
    
    
    dataloader_val = DataLoader(val_dataset, 
                                batch_size=config['batch_size'],
                                num_workers=2,
                                shuffle=False, # Important since you want to recreate the final big image 
                                pin_memory=True, 
                                drop_last=False)
    
    # Build the model
    model = build_model(config)
    
        
    # Print number of millions of parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Number of millions of parameters: {num_params}")
    # print number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Number of millions of trainable parameters: {num_trainable_params}")
    # send model to GPU
    model = model.cuda()
    
    
    
    # Get the criterion
    loss_configs = config['loss']
    criterion = get_loss_function(loss_configs)
    
    
    # Prepare training arguments
    warmup_epochs = config["warmup_epochs"]
    total_epochs = config["total_epochs"]
    warmup_ratio = warmup_epochs / total_epochs
    model_name = config["model"]["name"]
    model_output_dir = os.path.join(work_dir, f"{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    training_args = TrainingArguments(
        metric_for_best_model="fbeta",
        lr_scheduler_type="cosine",
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        learning_rate=float(config["lr"]),
        per_device_train_batch_size=config["batch_size"],
        load_best_model_at_end=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=total_epochs,
        save_total_limit=config["save_total_limit"] if "save_total_limit" in config else 10,
        report_to=config["report_to"],
        output_dir=str(model_output_dir),
        gradient_accumulation_steps=config["gradient_accumulation_steps"] if "gradient_accumulation_steps" in config else 1,
    )
    


    # Initialize Trainer
    trainer = VesuviusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VesuviusDataCollator(),
        criterion=criterion,
        eval_on_n_batches=config["eval_on_n_batches"],
    )
    
    # Perform training
    trainer.train()
    
    
        
        
        
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--work-dir", type=str, required=True, help="Path to the working directory")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the data directory")
    _args, _ = parser.parse_known_args()
    main(_args.config, _args.work_dir, _args.data_dir)
