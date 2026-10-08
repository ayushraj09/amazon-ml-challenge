# torchrun launcher: one cross-fitted fold, data-parallel over the visible GPUs. Usage: torchrun ... xenc_train_ddp.py <fold>
import sys
from src import xenc
xenc.train_models(folds=(int(sys.argv[1]),))
