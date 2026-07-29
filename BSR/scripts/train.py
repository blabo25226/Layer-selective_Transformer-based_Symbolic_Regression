import os
import glob
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pathlib import Path
import tempfile  # Import tempfile for temporary directory
import logging
from typing import Optional
from dataclasses import dataclass

from src.formula import create_both_vocabs, FormulaDataset
from src.transformer import LtnTransformer, create_callbacks, StopAfterDecay
from src.ConfigClasses import ConfigFormula, ConfigTransformer

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    run_name: str
    transformer_config: Optional[str]
    formula_config: Optional[str]
    base_dir: str
    batch_size: int = 128
    acc_grad_batches: int = 1
    backup_every: int = 1000
    top_k: int = 3
    num_workers: int = 96
    device: int = 0

def setup_logging(run_dir: str) -> None:
    """Configure logging for the training run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{run_dir}/train.log"),
            logging.StreamHandler()
        ]
    )

def main():
    parser = ArgumentParser(description="Train the Boolformer model with specified configurations.")
    parser.add_argument("-r", "--run_name", type=str, required=True, help="Name of the run for logging and checkpoints.")
    parser.add_argument("-t", "--transformer_config", type=str, help="Path to the Transformer configuration file.")
    parser.add_argument("-f", "--formula_config", type=str, help="Path to the Formula configuration file.")
    parser.add_argument("--bd", type=str, required=True, help="Base directory for storing checkpoints and backups.")
    parser.add_argument("--bs", type=int, default=128, help="Batch size of the training.")
    parser.add_argument("--agb", type=int, default=1, help="Accumulate grad batches")
    parser.add_argument("--be", type=int, default=1000, help="Backup every steps.")
    parser.add_argument("-k", type=int, default=3, help="Number of top backups to save.")
    parser.add_argument("--nw", type=int, default=96, help="Number of new workers.")
    parser.add_argument("-d", type=int, default=0, help="Cuda device")

    args = parser.parse_args()

    config = TrainingConfig(
        run_name=args.run_name,
        transformer_config=args.transformer_config,
        formula_config=args.formula_config,
        base_dir=args.bd,
        batch_size=args.bs,
        acc_grad_batches=args.agb,
        backup_every=args.be,
        top_k=args.k,
        num_workers=args.nw,
        device=args.d
    )

    # Create directories
    run_dir = os.path.join(config.base_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(run_dir)

    logging.info(f"Starting training run: {config.run_name}")

    # WandB Logger Initialization
    wandb_logger = WandbLogger(
        project='Boolformer',
        name=config.run_name,
        log_model=False,  # Disable model logging to wandb
        save_dir=tempfile.gettempdir(),  # Use a temporary directory for wandb files
        mode='online'  # Set to 'offline' if you don't want to sync with wandb server
    )

    # Define the directories for this run
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    backups_dir = os.path.join(run_dir, 'backups')

    # Check for existing last backup in the folder
    backups_pattern = os.path.join(backups_dir, "*.ckpt")
    backups = sorted(glob.glob(backups_pattern), key=os.path.getmtime, reverse=True)

    # Dataset setup
    c_formula = ConfigFormula(py_config_path=config.formula_config)

    # First create the model
    if len(backups) > 0:
        latest_backup = backups[0]
        print(f"Found backup {latest_backup}, resuming training...")
        ckpt_path = latest_backup
    else:
        print("No backup found, starting fresh training.")
        ckpt_path = None

    # Model Setup
    if ckpt_path:
        model = LtnTransformer.load_from_checkpoint(ckpt_path)
        c_transformer = model.c_transformer
    else:
        # Initialize a new model if no backups are found
        input_vocab, output_vocab = create_both_vocabs(config=c_formula)
        c_transformer = ConfigTransformer(py_config_path=config.transformer_config)
        model = LtnTransformer(c_transformer=c_transformer, input_vocab=input_vocab, output_vocab=output_vocab)

    # Now create callbacks after model is initialized
    callbacks = create_callbacks(config.base_dir, config.run_name, config.backup_every, config.top_k)
    # Add StopAfterDecay callback
    stop_callback = StopAfterDecay(total_steps=model.total_steps)
    callbacks.append(stop_callback)

    # Trainer Setup
    trainer = Trainer(
        default_root_dir=checkpoints_dir,       # Logs and checkpoints saved here
        accumulate_grad_batches=config.acc_grad_batches,    # The number of batches to accumulate
        logger=wandb_logger,                    # WandB logging
        callbacks=callbacks,                    # Updated to include all callbacks
        devices=[config.device],
        max_steps=model.total_steps  # Add this to ensure training stops after decay
    )

    # Load the Training Dataset
    train_dataset = FormulaDataset(input_vocab=model.input_vocab, 
                                   output_vocab=model.output_vocab,
                                   configFormula=c_formula,
                                   )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Start or resume training
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
