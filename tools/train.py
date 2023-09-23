from loguru import logger
from datetime import datetime
import argparse
import json
import yaml
from torch.utils.data import DataLoader

from kaggle_vesuvius.dataloaders import VesuviusDataset
from kaggle_vesuvius.models import Resnet3DSegModel

def main(config_path: str,
         work_dir: str,
         data_dir: str):
    
    with open(config_path, "rb") as f:
        config = yaml.load(f, yaml.FullLoader)
    
    # Print the config    
    logger.info(json.dumps(config, indent=4))
    return
    
    
    # Create datasets and  dataloaders
    train_dataset = VesuviusDataset(mode='train',
                                          data_dir=data_dir, 
                                          crop_size=config['crop_size'],
                                          eval_on=config['eval_on'],
                                          z_start=config['z_start'],
                                          z_end=config['z_end'],
                                          )
    
    val_dataset = VesuviusDataset(mode='val',
                                          data_dir=data_dir, 
                                          crop_size=config['crop_size'],
                                          eval_on=config['eval_on'],
                                          z_start=config['z_start'],
                                          z_end=config['z_end'],
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
    
    # Define the model
    assert config['model']['name'] in ['Resnet3DSegModel'], f"Model {config['model']['name']} not implemented"
    
    if config['model']['name'] == 'Resnet3DSegModel':
        model = Resnet3DSegModel(resnet_depth=config['model']['depth'])
        
    # Print number of millions of parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Number of millions of parameters: {num_params}")
    # print number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Number of millions of trainable parameters: {num_trainable_params}")
    # send model to GPU
    # model = model.cuda()
    
    
    # Get the criterion
    loss_configs = config['loss']
    criterion = get_loss_function(loss_configs)
    
    
    # Set training arguments
    warmup_epochs = config["warmup_epochs"]
    total_epochs = config["total_epochs"]
    warmup_ratio = warmup_epochs / total_epochs
    training_args = TrainingArguments(
        metric_for_best_model="fbeta",
        lr_scheduler_type="cosine",
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        learning_rate=float(config["lr"]),
        per_device_train_batch_size=1,
        load_best_model_at_end=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=50,
        per_device_eval_batch_size=2,
        num_train_epochs=total_epochs,
        save_total_limit=config["save_total_limit"] if "save_total_limit" in config else 10,
        report_to=config["report_to"],
        output_dir=str(model_output_dir),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
    )
    
    # Initialize the callback
    save_best_threshold_callback = SaveBestThresholdCallback()

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        compute_metrics=compute_fbeta_score,
        callbacks=[save_best_threshold_callback]
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
