# torchrun launcher: resume one fold from its checkpoint (ER_XENC_START_STEP / ER_XENC_LR set by the caller). Usage: ... <fold>
import sys
from src import xenc
xenc.train_models(folds=(int(sys.argv[1]),), resume=True)
