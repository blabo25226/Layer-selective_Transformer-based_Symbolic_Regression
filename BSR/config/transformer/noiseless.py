# Transformer parameters
D_EMBED = 4                     # Embedding dimension
D_MODEL = 512                   # Model dimension
NUM_ENCODER_LAYERS = 8          # Number of encoder layers
NUM_DECODER_LAYERS = 8          # Number of decoder layers
NUM_HEADS = 16                  # Number of attention heads
DIM_FEEDFORWARD = 2048          # Feedforward dimension size
DROPOUT = 0                     # Dropout rate
ATTENTION_SIZE = 199            # = Expression size - 1
INPUT_POINT_DIM_MAX = 10        # = Dimension max 
NB_INPUT_TOKENS = 512           # = 2**(dim max -1)

# Learning rate and scheduling
LEARNING_RATE = 0.0002          # Base learning rate
WARMUP_STEPS = 5000             # Number of warmup steps
STATIONARY_STEPS = 60000        # Number of steps at full learning rate
DECAY_STEPS = 5000              # Number of decay steps
WARMUP_START_FACTOR = 0.0005    # Initial learning rate factor during warmup
DECAY_END_FACTOR = 0            # Final learning rate factor after decay

#####################
## 59 M parameters ##
#####################