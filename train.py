import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import wandb
import torch
from torch.utils.data import Dataset, DataLoader


# Custom imports
from dataloaders import dataloader_ds1, dataloader_ds2
from models import unet
from eval import evaluate


# Use this function to print and log messages simultaneously
def print_and_log(msg):
    print(msg)
    logging.info(msg)
    # Flush the logging buffer
    logging.getLogger().handlers[0].flush()

def main():
    parser = argparse.ArgumentParser(description="Trains a UNet model")

    # Set up argparse arguments
    parser.add_argument('--in_channels', type=int, default=16, help='Input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Output channels')
    parser.add_argument('--init_features', type=int, default=64, help='Initial features')
    parser.add_argument('--class_one_weight', type=float, default=1., help='Weight for class one')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_steps', type=int, default=100000, help='Number of training steps')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency of logging and saving the model')
    parser.add_argument('--eval_steps', type=int, default=100, help='Number of evaluation steps on which to average the test metrics')
    parser.add_argument('--dataset_dir', type=str, default='../../datasets/dataset-1', help='Path to the dataset directory')
    parser.add_argument('--dataset_loader', type=str, default='dataloader_ds1', help='Which dataloader to use')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--cache_refresh_interval', type=int, default=64, help='Cache refresh interval')
    parser.add_argument('--cache_n_images', type=int, default=64, help='Number of images to cache')
    parser.add_argument("--save_model", action='store_true', default=False, help="Save the model")
    parser.add_argument("--wandb", action='store_true', help="Log to WandB")

    args = parser.parse_args()

    # Convert args to dict
    args = vars(args)
    print(args)
    
    print_and_log(f"[LOG] Args loaded (dict form): {args}")
    for key, value in args.items():
        print_and_log(f"[LOG] {key}: {value}")

    # Set the device
    if args['device'] == 'cuda' or args['device'] == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
        
    assert args['dataset_loader'] in ['dataloader_ds1', 'dataloader_ds2'], "dataset_loader must be either 'dataloader_ds1' or 'dataloader_ds2'"
        
    # Load datasets
    if args['dataset_loader'] == 'dataloader_ds1':
        train_dataset = dataloader_ds1.ImageSegmentationDataset(root=args['dataset_dir'], mode='train', device=device, cache_refresh_interval=args['cache_refresh_interval'], cache_n_images=args['cache_n_images'])
        test_dataset = dataloader_ds1.ImageSegmentationDataset(root=args['dataset_dir'], mode='test', device=device, cache_refresh_interval=args['cache_refresh_interval'], cache_n_images=args['cache_n_images'])
    if args['dataset_loader'] == 'dataloader_ds2':
        train_dataset = dataloader_ds2.ImageSegmentationDataset(root=args['dataset_dir'], mode='train', device=device, cache_refresh_interval=args['cache_refresh_interval'], cache_n_images=args['cache_n_images'])
        test_dataset = dataloader_ds2.ImageSegmentationDataset(root=args['dataset_dir'], mode='test', device=device, cache_refresh_interval=args['cache_refresh_interval'], cache_n_images=args['cache_n_images'])

    train_dataloader =  DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_dataloader =  DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True)

    # Create a model
    print_and_log("Creating a model...")
    model = unet.UNet(in_channels=args['in_channels'], out_channels=args['out_channels'], init_features=args['init_features'], class_one_weight=args['class_one_weight'])
    # send to device
    model.to(device)
    print_and_log("Model created.")
    
    
    # Set up wandb
    if args['wandb']:
        wandb.login(key="YOUR_WANDB_API_KEY")
        wandb.init(project='vesuvius-challenge', config=args)
    
    
    # Train the model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    # Initialize the metrics dictionary
    metrics = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "f0point5": []
    }
    
    # Initialize a DataFrame to store the metrics
    metrics_df = pd.DataFrame(columns=["step"] + list(metrics.keys()))

    print_and_log("Starting training...")
    for step in tqdm(range(args['num_steps'])):
        # refresh the cache 
        if step % args['cache_refresh_interval'] == 0:
            train_dataset.reset()
            test_dataset.reset()
            
            # Re-initialize the dataloaders
            train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True)
        
        
        # Forward pass and compute the loss
        batch = next(iter(train_dataloader))
        images, labels = batch['image'], batch['targets']
        # send to device
        images = images.to(device)
        labels = labels.to(device)
        
        images = images.permute(0, 3, 1, 2)
    
        optimizer.zero_grad()
        out = model(images, labels)
        loss = out['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update the metrics
        for key in metrics.keys():
            metrics[key].append(out[key].detach().cpu().numpy())

        # Logging and saving
        if step % args['log_freq'] == 0:
            
            
            # Merge all numbers from each array to a numpy array
            for key in metrics.keys():
                metrics[key] = [value.item() for value in metrics[key]]
            
            # Calculate average metrics
            avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
            
            # Get test metrics
            avg_test_metrics = evaluate(model, test_dataloader, eval_steps=args['eval_steps'], device=device)
            
            # combine the metrics
            avg_metrics = {**avg_metrics, **avg_test_metrics}
            
            # Log metrics to wandb
            if args['wandb']:
                wandb.log(avg_metrics, step=step)

            # Log metrics to a DataFrame
            avg_metrics["step"] = step
            avg_metrics_df = pd.DataFrame([avg_metrics])
            metrics_df = pd.concat([metrics_df, avg_metrics_df], ignore_index=True)

            # Save the metrics DataFrame as a CSV file
            metrics_df.to_csv("metrics.csv", index=False)

            # Reset the metrics
            for key in metrics.keys():
                metrics[key] = []
                
            

        # Save the model
        if step % args['log_freq'] == 0 and args['save_model']:
            if "models" not in os.listdir():
                os.mkdir("models")
            
            torch.save(model.state_dict(), f"models/model_step={step}.pth")

    # Close the wandb run
    if args['wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()
