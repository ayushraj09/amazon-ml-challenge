# Licences of models and dependencies

## Pretrained model weights
The challenge requires MIT or Apache-2.0 licences and at most 8B parameters.

| Model | Role | Licence | Params | Pinned revision |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | bi-encoder for candidate retrieval (fine-tuned) | MIT | 118M | 614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| `jhu-clsp/mmBERT-small` | listwise cross-encoder (fine-tuned) | MIT | 141M | abc32620dd4f6ab06f5fbe905dc25f310618e09f |
| `Qwen/Qwen3-1.7B` | second listwise cross-encoder (LoRA fine-tuned), final ensemble | Apache-2.0 | 1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| `Qwen/Qwen3-4B` | third listwise cross-encoder (LoRA fine-tuned), final ensemble G2 / G3 | Apache-2.0 | 4B | 1cfa9a7208912126459214e8b04321603b3df60c |

- Licences were checked on the Hugging Face API (licence tag) on 2026-09-25.
- The weights are downloaded once from the Hugging Face Hub by `transformers`, a model-hub download.
- No business data, lookup or external dataset is fetched.

## Python dependencies (pinned in `requirements.txt`)
| Package | Licence |
|---|---|
| polars | MIT |
| pyarrow | Apache-2.0 |
| numpy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scipy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| rapidfuzz | MIT |
| lightgbm | MIT |
| xgboost | Apache-2.0 |
| catboost | Apache-2.0 |
| sparse-dot-topn | Apache-2.0 |
| torch | BSD-3-Clause |
| transformers | Apache-2.0 |
| tokenizers | Apache-2.0 |
| safetensors | Apache-2.0 |
| huggingface-hub | Apache-2.0 |
| sentencepiece | Apache-2.0 |
| sentence-transformers | Apache-2.0 |
| accelerate | Apache-2.0 |
| peft | Apache-2.0 |
| cython | Apache-2.0 |
| numba | BSD-2-Clause |
| joblib | BSD-3-Clause |
| libmr | BSD-3-Clause (see [Vastlab/libMR LICENSE.TXT](https://github.com/Vastlab/libMR/blob/master/LICENSE.TXT)) |
| pytest | MIT |

## Reference implementations (ideas and hyperparameters reimplemented, no code vendored)
- OP_Match
- IOMatch / OpenLTR
- open-set-biometrics (ECCV'24)
- NIL_EL
- CaLiGraph / NASTyLinker
- ExtremeValueMachine
- OSDN (OpenMax)
- Ditto
- BLINK
- kaggle_foursquare (4th place)

Each is cited at the point of use in the code and in `IMPLEMENTATION_PLAN.md` §13.
