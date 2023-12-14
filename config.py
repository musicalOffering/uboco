# DATASET_TYPE = "TSN"
DATASET_TYPE = "RES"
if DATASET_TYPE == "TSN":
	DATA_ROOT = './data/TSN_4fps_pad/' # for TSN
elif DATASET_TYPE == "RES":
	DATA_ROOT = './data/resnet50_padded/' # for ResNet50

DATASET_MODE = "tv" # "tvt", "tv", "vtt"

if DATASET_MODE == "tv":
	# TRAIN / VAL
	TRAIN_ANNOTATION_PATH = f'{DATA_ROOT}k400_mr345_train_min_change_duration0.3.pkl'
	VAL_ANNOTATION_PATH = f'{DATA_ROOT}k400_mr345_val_min_change_duration0.3.pkl'
	TEST_ANNOTATION_PATH = f'{DATA_ROOT}k400_mr345_val_min_change_duration0.3.pkl'
	# TEST_ANNOTATION_PATH = 'test_len.json'
	TRAIN_DATA_PATH = f'{DATA_ROOT}train_of_train/'
	VAL_DATA_PATH = f'{DATA_ROOT}val/'
	TEST_DATA_PATH = f'{DATA_ROOT}val/'

elif DATASET_MODE == "tvt":
	# TRAIN_of_TRAIN / VAL_of_TRAIN / VAL
	TRAIN_ANNOTATION_PATH = f'{DATA_ROOT}k400_mr345_train_min_change_duration0.3.pkl'
	VAL_ANNOTATION_PATH = f'{DATA_ROOT}k400_mr345_train_min_change_duration0.3.pkl'
	TEST_ANNOTATION_PATH = f'{DATA_ROOT}k400_mr345_val_min_change_duration0.3.pkl'
	# TEST_ANNOTATION_PATH = 'test_len.json'
	TRAIN_DATA_PATH = f'{DATA_ROOT}train_of_train/'
	VAL_DATA_PATH = f'{DATA_ROOT}val_of_train/'
	TEST_DATA_PATH = f'{DATA_ROOT}val/'

elif DATASET_MODE == "vtt":
	# TRAIN_of_TRAIN / VAL_of_TRAIN / VAL
	TRAIN_ANNOTATION_PATH = f'{DATA_ROOT}k400_mr345_train_min_change_duration0.3.pkl'
	VAL_ANNOTATION_PATH = f'{DATA_ROOT}k400_mr345_train_min_change_duration0.3.pkl'
	TEST_ANNOTATION_PATH = f'{DATA_ROOT}k400_mr345_val_min_change_duration0.3.pkl'
	# TEST_ANNOTATION_PATH = 'test_len.json'
	TRAIN_DATA_PATH = f'{DATA_ROOT}val_of_train/'
	VAL_DATA_PATH = f'{DATA_ROOT}train_of_train/'
	TEST_DATA_PATH = f'{DATA_ROOT}val/'


PRED_PATH = 'k400_pred.pkl'
VISUAL_DATA_PATH = './resnet_visualizing/'
# VISUAL_DATA_PATH = './visualizing/'

SAVE_MODEL = False
MODEL_SAVE_PATH = './models/'
SAVE_RESULT = False
RESULT_PATH = './results'

VAL_THRESHOLD = 0.05

DEVICE = 'cuda'

if DATASET_TYPE == "RES":
	FEATURE_DIM = 2048 # 2048 for Resnet, 4096 for TSN
elif DATASET_TYPE == "TSN":
	FEATURE_DIM = 4096

FEATURE_LEN = 40 # DO NOT CHANGE
TIME_UNIT = 0.25 # DO NOT CHANGE

GAP = 8 # VALID LOCAL RANGE
CHANNEL_NUM = 64 
ENCODER_HIDDEN = 768
LONG_LAYER_NUM = 4
DECODER_HIDDEN = -1

AUX_LOSS_COEF = 0.5
L1_COEF = 0.

BATCH_SIZE = 32
LEARNING_RATE = 1e-3

if DATASET_MODE == 'tv':
	EPOCHS = 100
	PATIENCE = 10 # Patience for early stopping	
elif DATASET_MODE in ['tvt', 'vtt'] :
	EPOCHS = 100
	PATIENCE = 10 # Patience for early stopping

GLUE_PROB = 0.0 # Probability of glueing augmentation 

VAL_VIDEOS = ["01BFInmg3Zs_val", "77aDh42ddw8_val", "_4insWyfuuw_train", "_4oxxQeQ2aM_train", "hlUDq3QKZo8_val"] # files in visualizing folder

SIGMA = 0.5
GAUSSIAN_KERNEL = 3
MAX_POOL_KERNEL = 5
# Minimum score to be event boundary

UNIT_SQUARE_SIZE = 9
MINIMUN_SQUARE_SIZE = 5
TOPK_RATIO = 0.2
#MIN_STD = 0.35
MIN_DIFF = 1.
BIG_MINUS = -100.
 
ver = f"recent" # customized ver string
VER = f'{ver} DATASET({DATASET_TYPE}) GAP({GAP})) ({DATASET_MODE})'

NUM_WORKERS = 10

