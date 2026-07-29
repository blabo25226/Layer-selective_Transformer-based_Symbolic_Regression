import os
import heapq
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import collections  # Add this import
from pytorch_lightning.utilities import rank_zero_info

# Function to create necessary directories
def create_run_directories(base_dir, run_name):
    run_dir = os.path.join(base_dir, run_name)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    backups_dir = os.path.join(run_dir, 'backups')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(backups_dir, exist_ok=True)
    
    return checkpoints_dir, backups_dir


# Custom callback to save backups every N steps
class SaveBackupsEveryNSteps(Callback):
    def __init__(self, interval, dirpath, top_k=3, backup_every=1000):
        """
        Initializes the callback to save backups every 'interval' steps and keep top_k based on averaged loss.

        Args:
            interval (int): The interval in steps at which to save backups.
            dirpath (str): Directory where backups will be saved.
            top_k (int): Number of top backups to keep based on averaged loss.
            backup_every (int): Number of steps over which to average the loss.
        """
        super().__init__()
        self.interval = interval 
        self.dirpath = dirpath
        self.top_k = top_k
        self.backup_every = backup_every
        os.makedirs(self.dirpath, exist_ok=True)

        # Use negative loss values so the heap keeps highest losses at the top
        self.backup_heap = []  # Stores tuples of (-average_loss, ckpt_path)
        self.loss_queue = collections.deque(maxlen=self.backup_every)
        
        # Scan directory for existing checkpoints
        self._initialize_from_existing_checkpoints()
        
    def _initialize_from_existing_checkpoints(self):
        """Scan the checkpoint directory and initialize the heap with existing checkpoints."""
        if not os.path.exists(self.dirpath):
            return
        
        existing_checkpoints = []
        rank_zero_info(f"Scanning directory {self.dirpath} for existing checkpoints...")
        
        for filename in os.listdir(self.dirpath):
            if filename.endswith('.ckpt'):
                ckpt_path = os.path.join(self.dirpath, filename)
                try:
                    # Extract average loss from filename using regex
                    import re
                    # Try both patterns: avg_loss and loss, and handle potential trailing period
                    avg_loss_match = re.search(r'avg_loss=([0-9.]+)', filename)
                    loss_match = re.search(r'loss=([0-9.]+)', filename)
                    
                    if avg_loss_match:
                        # Remove any trailing period from the matched value
                        avg_loss_str = avg_loss_match.group(1).rstrip('.')
                        avg_loss = float(avg_loss_str)
                        rank_zero_info(f"Found checkpoint with avg_loss={avg_loss}: {filename}")
                        existing_checkpoints.append((-avg_loss, ckpt_path))
                    elif loss_match:
                        # Remove any trailing period from the matched value
                        loss_str = loss_match.group(1).rstrip('.')
                        loss = float(loss_str)
                        rank_zero_info(f"Found checkpoint with loss={loss}: {filename}")
                        existing_checkpoints.append((-loss, ckpt_path))
                    else:
                        rank_zero_info(f"Skipping checkpoint with no loss info: {filename}")
                except (ValueError, IndexError) as e:
                    rank_zero_info(f"Error parsing checkpoint {filename}: {str(e)}")
                    continue
        
        rank_zero_info(f"Found {len(existing_checkpoints)} checkpoints with loss information")
        
        if existing_checkpoints:
            # Sort by loss (lowest first since we're using negative values)
            existing_checkpoints.sort()
            rank_zero_info(f"Sorted checkpoints (best first): {[path.split('/')[-1] for _, path in existing_checkpoints]}")
            
            # Initialize the heap with the best checkpoints
            self.backup_heap = []
            for i, (neg_loss, path) in enumerate(existing_checkpoints):
                if i < self.top_k:
                    heapq.heappush(self.backup_heap, (neg_loss, path))
                    rank_zero_info(f"Keeping checkpoint: {path.split('/')[-1]} with loss={-neg_loss}")
                else:
                    # Remove excess checkpoints
                    if os.path.exists(path):
                        rank_zero_info(f"Removing excess checkpoint: {path.split('/')[-1]} with loss={-neg_loss}")
                        os.remove(path)
                    
        rank_zero_info(f"Initialized checkpoint heap with {len(self.backup_heap)} existing checkpoints")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        """
        Called at the end of every batch. Saves a backup if the current step matches the interval.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The LightningModule being trained.
            outputs (Any): The outputs of the training step.
            batch (Any): The current batch.
            batch_idx (int): The index of the current batch.
            unused (int): Unused argument for compatibility.
        """
        global_step = trainer.global_step
        train_loss = float(trainer.callback_metrics.get("train_loss"))

        if train_loss is not None:
            self.loss_queue.append(train_loss)

        # Save checkpoint every 'interval' steps
        if global_step > 0 and global_step % self.interval == 0 and len(self.loss_queue) == self.backup_every:
            # Calculate the average loss over the last 'backup_every' steps
            average_loss = sum(self.loss_queue) / len(self.loss_queue)

            # Create the checkpoint filename with step and averaged loss
            filename = f"step={global_step}-avg_loss={average_loss:.6f}.ckpt"
            ckpt_path = os.path.join(self.dirpath, filename)
            
            # Save the checkpoint
            trainer.save_checkpoint(ckpt_path)

            # Store negative loss so highest losses are at top of heap
            heapq.heappush(self.backup_heap, (-average_loss, ckpt_path))

            # If heap exceeds top_k, remove the checkpoint with highest loss
            if len(self.backup_heap) > self.top_k:
                _, worst_ckpt_path = heapq.heappop(self.backup_heap)
                if os.path.exists(worst_ckpt_path):
                    os.remove(worst_ckpt_path)
                    # rank_zero_info(f"Removed checkpoint {worst_ckpt_path} as it had higher loss than top-{self.top_k}")
            
            # print(f"Backup checkpoint saved at step {global_step} with average loss {average_loss:.6f} in {ckpt_path}")

# New Callback to keep only the latest backup in the backups directory
class SaveLatestBackup(Callback):
    def __init__(self, dirpath, interval):
        """
        Initializes the callback to keep only the latest backup in the directory.

        Args:
            dirpath (str): Directory where the latest backup will be saved.
            interval (int): The interval in steps at which to save backups.
        """
        super().__init__()
        self.dirpath = dirpath
        self.interval = interval
        os.makedirs(self.dirpath, exist_ok=True)
        self.latest_backup = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        global_step = trainer.global_step
        
        # Only save backup at the specified interval
        if global_step > 0 and global_step % self.interval == 0:
            # Get the current loss
            train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))
            backup_filename = f"step={global_step}-loss={train_loss:.6f}.ckpt"
            backup_path = os.path.join(self.dirpath, backup_filename)
            
            # Save the latest backup
            trainer.save_checkpoint(backup_path)

            # Remove the previous backup if it exists
            if self.latest_backup and os.path.exists(self.latest_backup):
                os.remove(self.latest_backup)
            
            self.latest_backup = backup_path

# Add this new callback
class StopAfterDecay(Callback):
    """Callback to stop training after the learning rate decay phase is complete."""
    def __init__(self, total_steps):
        """
        Args:
            total_steps (int): Total number of steps after which to stop 
                             (warmup + stationary + decay steps)
        """
        super().__init__()
        self.total_steps = total_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step >= self.total_steps:
            rank_zero_info(f"\nStopping training after completing {self.total_steps} steps (decay phase complete)")
            trainer.should_stop = True

# Factory function to create both callbacks
def create_callbacks(base_dir, run_name, backup_every, top_k=3):
    """
    Creates the necessary callbacks for checkpointing and backups.

    Args:
        base_dir (str): Base directory for the run.
        run_name (str): Name of the current run.
        backup_every (int): Interval in steps to save backups.
        top_k (int): Number of top backups to keep.

    Returns:
        list: List of callback instances.
    """
    checkpoints_dir, backups_dir = create_run_directories(base_dir, run_name)
    
    backups_callback = SaveBackupsEveryNSteps(
        interval=backup_every,
        dirpath=checkpoints_dir,
        top_k=top_k,
        backup_every=backup_every
    )
    
    latest_backup_callback = SaveLatestBackup(
        dirpath=backups_dir,
        interval=backup_every
    )
    
    return [backups_callback, latest_backup_callback]