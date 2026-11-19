"""Hyperparameters shared by the model, training and benchmark scripts."""

# Model shape
D_MODEL = 32                    # embedding width of every token
N_HEADS = 4                     # number of attention heads
D_HEAD = D_MODEL // N_HEADS     # width of one head
FFN_EXPANSION = 4               # hidden width multiplier of the feed-forward layer
D_FF = D_MODEL * FFN_EXPANSION  # hidden width of the feed-forward layer
N_LAYERS = 2                    # number of transformer blocks
BLOCK_SIZE = 32                 # context length (rows of the position table)
EPS = 1e-5                      # keeps RMSNorm away from division by zero
INIT_STD = 0.02                 # std of the random weight initialisation

# Data
DATA_PATH = "data/input.txt"
TRAIN_RATIO = 0.9               # first 90% for training, last 10% for validation
BATCH_SIZE = 16
RANDOM_SEED = 0

# Training
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.01
N_STEPS = 2000
EVAL_EVERY = 200                # evaluate train / val loss every N steps
EVAL_ITERS = 20                 # batches averaged per evaluation
WARMUP_STEPS = 100              # linear learning-rate warm-up length
MAX_GRAD_NORM = 1.0             # gradient clipping threshold

# Paths
CHECKPOINT_PATH = "checkpoints/mini_gpt.pt"
ASSETS_DIR = "assets"
