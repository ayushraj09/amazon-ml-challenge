# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Mayfair
**Team Members:** Ayush Raj, Kesav Patneedi, Kunal Sewal (Indian Institute of Technology (IIT), Bhilai)
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

- **Pipeline.** Four stages:
  1. A **cross-fitted multilingual bi-encoder** retrieves candidates, together with two lexical channels.
  2. A **cheap learned filter** keeps only each record's two best candidates. The result is `candidate_pairs.tsv`: **4.89 candidates per Source-1 entity on test**, a reduction of more than 99.999% over all within-country pairs.
  3. A gradient-boosted matcher, stacked with **listwise cross-encoders that have an explicit "no match" option**, scores every candidate. The final model is a weighted average of four such stacks: `mmBERT-small`, **Qwen3-1.7B + LoRA**, a Qwen3-1.7B continued with French-filler augmentation, and **Qwen3-4B + LoRA**.
  4. Each Source-1 entity's final match set is the one that **maximises expected F0.5**.
- **Key findings.**
  - The data generator plants **near-miss distractors** (legal form swapped, house number shifted by 1–50) and **sibling distractors** (same first word and same address, but a different real category word).
  - The test set adds a country, **France**, that is absent from train.
- **What mattered most.** Our first submission scored 0.98030 on the public leaderboard. A label-free diagnosis traced the gap to France, which is 15% of test, and treated it as a domain-generalisation problem:
  - transfer-robust models are routed to countries unseen in training;
  - a language-independent vocabulary of "filler" words is mined from each country's own data;
  - a real-word vocabulary feature separates sibling businesses from typos;
  - leave-one-country-out training is used as the evaluator;
  - three **unseen-country guards**, each derived from train truth rates, drop generator distractor patterns that the models accept in France far more often than in train.

  **Public leaderboard: 0.98030 (v1) → 0.98830 (v4) → 0.99001 (E2h) → 0.99008 (E5a) → 0.99054 (G3) → 0.99065 (final).** The final submission adds, on top of v4, the unseen-country guards, French-format and filler augmentation, the Qwen3-4B cross-encoder and a validated 4-model weighted ensemble.

---

## 2. Methodology

### 2.1 Problem Analysis
All numbers were measured on the provided data; details are in `FINDINGS.md`.

**Structure**
- Every S2/S3 record matches **at most one** S1, and no match crosses countries.
- Matches per S1: mean 3.46; 5.6% of S1s are singletons.
- About 26% of train S2/S3 records match nothing.
- Matching is therefore framed **record-centrically**, as open-set 1:N identification: each record either belongs to one S1 or to none.

**Noise operators (matches)**
- Names:
  - legal suffix dropped, changed or bracketed;
  - Indic-script transliterations, accents, leetspeak;
  - domain names, word reorder;
  - `dba` / `fka` aliases and acronyms;
  - invented brand names;
  - **category word replaced by a generic filler** ("Ridge Pediatrics Inc" → "Ridge Inc Services").
- Addresses: abbreviations, state vs region vs department, number prefixes (`##7`, `H.no`, `PLOT 817`), digit drops, component reordering, missing address (4.4%).

**Distractor operators (non-matches)**, measured on train labels:

| Distractor | Signature | True-match rate in train |
|---|---|---|
| Near-miss | same name, legal form swapped, house number shifted 1–50 | 3% of true pairs have a shifted number vs 80% of look-alike distractors |
| **Sibling business** | same first word, same address, one category word replaced by a *different real word* | India 8.6%, US 4.2% (vs 90% / 53% when the changed word is a garble) |

**Train → test shift**
- About 1.1 more distractor records per S1 on test; near-miss look-alikes: US 0.30 → 0.55 per S1.
- **France** (test only) shows **10× more acronym names** and uses **French filler and category vocabulary**: "& Fils", "et Associés", "Groupe", "Développement".

### 2.2 Solution Strategy
**Approach type:** blocking plus a classifier cascade, with open-set identification and domain-generalisation components.

**Core ideas**
1. **Test-like validation (F_shift):** synthetic near-miss distractors at the measured test rates, a France proxy, and re-weighting to the test country mix.
2. **Leave-one-country-out (LOCO) evaluation for the unseen country:** train on US and score India, and the reverse. This measures what F_shift cannot see.
3. **Country routing:** full-feature models for countries in train; transfer-robust models for any other country.
4. **Label-free, language-independent vocabulary features**, mined per country from that country's own data.
5. **A small, learned candidate set**, selected on the size / recall / F0.5 frontier.
6. An exact per-entity **expected-F0.5** decision.

### 2.3 Validation protocol
- **Validation S1s:** 20% of train S1s (441,235), chosen by hash. All models use 5-fold GroupKFold by the record's true S1, so every validation pair is scored out-of-fold. The test models are the fold ensembles, which have together seen all of train.
- **F_val:** macro-F0.5 on real validation records.
- **F_shift:** adds synthetic near-miss distractors at test rates (US 0.55, India 0.17, France proxy 0.70) and re-weights to the test mix (US 38%, France 15%, India 47%).
- **LOCO:** the stage-2 model trained on one train country and scored on the other. It is used to choose the transfer feature set.
- **Robustness check "V1":** remove 15% of validation S1s from the gallery. The drop must be under 0.01; every version passed with |drop| ≤ 0.0001.

---

## 3. Candidate Generation (Blocking)

**Step 1: retrieval** (per country; the union of three channels)
1. **Dense (primary).** Fine-tuned `intfloat/multilingual-e5-small` bi-encoder (MIT, 118M parameters) with exact GPU top-50 by cosine similarity.
   - Training follows the BLINK recipe: lr 1e-5, batch 128, in-batch negatives drawn from a single country, scale 20.
   - **Cross-fitted** over 2 S1 folds.
2. **Word TF-IDF** on the normalised core name, top-20, for records without an address.
3. **Exact-key joins** on (house number, street word) and on sorted core-name tokens.

Retrieval gives about 70 candidates per record, about 50 after channel restriction, with a validation recall of **0.99753**.

**Step 2: learned candidate filter (stage 1)**, which produces `candidate_pairs.tsv`.
- **Model:** GPU XGBoost on 25 cheap features, with 5-fold out-of-fold predictions and the Foursquare 4th-place level-1 configuration.
- **Features:** retrieval scores and ranks, channel flags, name and street token-set similarity, house-number equality and difference, legal-form relation, list statistics.
- **Keep rule:** a record keeps **its 2 best candidates with stage-1 probability ≥ 0.01**. The organisers rank smaller candidate sets higher, so the rule was chosen on the measured frontier, **after retraining the matcher on each candidate set**:

| Keep rule | Candidates per val S1 | Candidates per test S1 | Recall ceiling (val) | Stage-2 F_shift after retraining |
|---|---|---|---|---|
| top-12 OR p ≥ 0.001, cap 16 (initial) | 57.0 | 69.3 | 0.99447 | 0.98552 ¹ |
| top-16 AND p ≥ 0.001 | 6.37 | 7.88 | 0.99429 | 0.98712 |
| **top-2 AND p ≥ 0.01 (final)** | **4.23** | **4.89** | **0.98604** | **0.98717** |

¹ Earlier feature set.

- Shrinking the candidate lists did not cost F0.5, and retraining on the shorter lists *raised* it: the list-level features and the listwise cross-encoder become sharper when the lists are short.
- The true pairs lost by the tighter rule are ones the matcher never selected. Almost all are records without an address whose names are shared by several S1s of a chain.

**Candidate pairs**

| | Train | Test |
|---|---|---|
| After retrieval (channel-restricted) | 522.6M | 499.8M |
| **`candidate_pairs.tsv`** | 9.3M (representatives) | **8,478,551 pairs = 4.89 per S1** |

- 1,731,879 of the 1,732,544 test S1s have at least one candidate.
- Per country: US 4.47, India 5.20, France 5.03.
- The matcher, the cross-encoder and the stacked models run **only** on this set.

---

## 4. Matching Model

### Architecture at a glance

```
 candidate pairs (4.89 per S1)
            │
            ▼
 ┌──────────────────────────────────────────────┐
 │ Stage 2: XGBoost on ~120 pair features  → p2 │
 └──────────────────────────────────────────────┘
            │  records with best p2 in (0.01, 0.99)
            ▼
 ┌────────────┬─────────────┬──────────────────┬─────────────┐
 │mmBERT-small│ Qwen3-1.7B  │ Qwen3-1.7B (v9,  │  Qwen3-4B   │   listwise cross-encoders
 │ 141M, MIT  │ LoRA, Apache│ + French fillers)│ LoRA, Apache│   with a "none of these" option
 └────────────┴─────────────┴──────────────────┴─────────────┘
            │  one stacked XGBoost per cross-encoder
            │  (stage-2 features + cross-encoder outputs + sibling consensus)
            ▼
 weighted average  (US/India 1.5/1.5/3/0.75 · France 1.5/2/3/1)
            ▼
 per-S1 expected-F0.5 decision  +  unseen-country guards  →  matching_results.tsv
```

### 4.1 Normalisation (all dictionaries mined from the provided data)
- Case, Latin accents, junk prefixes, leetspeak, duplicated words and honorifics are removed.
- Indic tokens are mapped through a dictionary mined from aligned train pairs (1,312 tokens).
- The legal form is split out and canonicalised with spelling variants mined from train.
- House-number, compound-number and unit parsing.
  - French prefixes `N° 117` and `(46)` were added after a qualitative audit of France test decisions. Before the fix, 67k France test records (4.6%) had no house number, and their street began with "°".
  - The fix changed 59,630 France candidate pairs from "number unknown" to "same number". It lets the sibling guard catch siblings written with `N°`.
  - `N°` never occurs in train, and `(n)`-first addresses occur in 76 of 10M train records, so the trained models were kept and only the test features were recomputed on the same candidate set.
- Per-country street and region maps: mined from train pairs, or for test-only countries from unlabelled test pseudo-pairs.

### 4.2 Pair features (stage 2)
- **Name:** rapidfuzz ratio, partial, token-sort, token-set, Jaro-Winkler, Levenshtein; no-space and domain-stem similarity; alias parts; token counts.
- **Acronym:** record acronym equality, and **record name = the S1's initials** (all tokens, and tokens of 3+ letters).
- **Near-miss block:** house-number equality, signed and absolute difference, digit-edit type, closest number, compound-number tuple, bis/ter, unit numbers, legal-form relation.
- **Address:** street and address token-set similarity, component overlap, state inclusion, missing-address flags.
- **Chain-ness:** how many S1s share the name.
- **Open-set features:**
  - libMR meta-recognition on the candidate score tail (tail 20);
  - maximum logit and margin (Vaze et al. 2022);
  - **Extreme Value Machine** inclusion probability (tail 50, margin scale 0.5);
  - competition ranks and gaps.
- **Filler-vocabulary features** (label-free, per country):
  - Among candidate pairs at the same address with the same first word, count the tokens only the record has.
  - A **filler** is frequent there (≥ 1% of those tokens) and rare in the clean S1 names (added-count / S1-name count ≥ 0.25).
  - Run without labels on train, the miner recovers exactly the fillers found in labelled true pairs: US and India `center, partners, service, services`. On test France it finds `and, associés, développement, et, fils, groupe, services`.
  - Features: all extra words are fillers; the record's non-filler words are all in the S1 name.
- **Real-word vocabulary features** (label-free, per country):
  - Are the record's extra words **real words of the country's clean S1 names** (frequency ≥ 3)? Also the log frequency of the rarest extra word.
  - Real-word swaps are sibling businesses; garbles are typos of the same business.
  - This rule is learned on train and carries over to French vocabulary without any French labels.

### 4.3 Models
- **Stage 2:** GPU XGBoost with the Foursquare 4th-place level-2 configuration (lr 0.05, depth 15, colsample 0.5, L1 1, L2 10, early stopping 100), 5-fold GroupKFold, easy negatives sub-sampled.
- **Cross-encoder:** listwise, on `jhu-clsp/mmBERT-small` (MIT, 141M parameters), chosen by a bake-off of 10 licence-compliant backbones.
  - Candidates plus a learned **NIL** option.
  - Losses: NIL softmax, OpenMatch one-vs-all with the hardest negative, relative threshold minimisation (weight 4), auxiliary number / legal / street heads (0.5 each).
  - Trained with lr 3e-5, warmup 0.1 and dev-AP early stopping; cross-fitted over 2 folds by owner S1.
  - Retrained on the final top-2 candidate lists: dev AP 0.9977 / 0.9982.
- **Stacked model:** XGBoost on the stage-2 features plus cross-encoder outputs (logit, one-vs-all score, list probability, fused score, rank and gap), rank-1 agreement, and **sibling consensus** (other records confidently assigned to the same S1). All inputs are out-of-fold on train.
- **Country routing:**
  - Two stage-2 and stacked models are trained: *full* (all features) and *transfer*. The transfer set drops `legal_pair_id`, the categorical identity of the legal-form pair, which is unseen for French pairs, and the record-style flags.
  - LOCO chose this set: mean US↔India transfer F0.5 **0.95995 → 0.96331**. Dropping the stage-1 score features (0.949) or the signed number shift (0.963) hurt, so they stay.
  - S1s of countries present in train use the full model (stacked F_shift 0.99130); any other country uses the transfer model (0.99074), each with its own validated decision rule.
- **Second cross-encoder: Qwen3-1.7B + LoRA** (Apache-2.0, 1.7B parameters, revision pinned).
  - The same listwise-with-NIL objective on the same top-2 candidate lists.
  - The prompt is `Business A: … / Business B: … / Same business:` with last-token pooling.
  - LoRA r 64, α 16, dropout 0.1, lr 2e-4, bf16, gradient checkpointing; up to 300k records per fold, 2 cross-fitted folds, dev-AP early stopping (dev AP 0.9984 / 0.9977).
  - Its own stacked models (full and transfer) use the same stage-2 features.
- **French-format augmentation** (`ER_XENC_AUG=0.5`, `cross_encoder._augment`), in a second Qwen3-1.7B run (v8). With probability 0.5, a training record and all its candidates are rewritten with one consistent random mapping:
  - US/India legal forms → French forms, with the variants of one form mapped to the same target, so a swap stays a swap and a same-form pair stays same-form;
  - "India"/"USA" → "France";
  - house-number prefixes → `N° 12` / `(12)` / `No 12` / `#12`.

  Labels are unchanged.
  - **Labelled check:** the same 3,000 held-out dev records per fold, rewritten into French formats. The original Qwen loses **0.0077 / 0.0094** dev AP; the augmented one loses **0.0005 / 0.0006**. Plain dev AP is unchanged (0.99811 / 0.99779 vs 0.99838 / 0.99774).
  - **Use:** France's transfer route averages mmBERT-small with this model (`--transfer-runs`). US and India keep mmBERT-small + the original Qwen, the best validated pair.
- **Filler augmentation** (run v9: `ER_XENC_AUG_FILL=1`, `cross_encoder._augment`). This continues training v8 on 100k other train records per fold (the same fold halves, so out-of-fold scores stay valid), with lr 1e-4 and its own best checkpoint.
  - The fillers mined from the train countries (center, partners, service(s)) are rewritten, consistently within each record group, into France's mined fillers (Groupe, Développement, & Associés, Et Fils, Services). A filler replacing or extending the category word at the same address is true in both train countries (US 0.995, India 0.978), so labels are unchanged.
  - **Labelled check** (the same dev records in French formats **with French fillers**): v8 loses **0.0085 / 0.0074** dev AP, v9 only **0.0006 / 0.0007**. Plain dev AP is 0.99831 / 0.99806 (v8: 0.99811 / 0.99779).
  - Averaged with mmBERT-small on France's transfer route (E6). US and India are unchanged.
- **Third cross-encoder: Qwen3-4B + LoRA** (run D; Apache-2.0, 4B, revision `1cfa9a72…`). Trained on a second machine with data parallelism (2 folds × 352k records, 2 GPUs per fold), with format augmentation.
  - Fold 1 diverged at lr 2e-4 and was resumed from its step-1,000 checkpoint at lr 1e-4.
  - Dev AP **0.99851 / 0.99774**; stacked F_shift full **0.99162**, transfer **0.99111**, the best single model on both routes.
  - Exact code and logs: `experiments/qwen4b_remote_training/`.
- **Ensemble choice by exhaustive validation:** all 31 subsets of the 5 stacked runs were scored per route. Every model has a positive average effect (Qwen3-4B +0.00014, mmBERT-small +0.00010–0.00011, Qwen3-1.7B +0.00009–0.00011, the two augmented 1.7B runs +0.00001–0.00003), so no model drags the average down. The top subsets are within 0.00001. G4 uses mmBERT-small + Qwen3-1.7B + Qwen3-4B + v9 on both routes: US/India 0.99176, France route 0.99126, vs 0.99171 / 0.99121 for G3.
- **Weights (final):** a coordinate search from equal weights on validation put the Qwen3-4B at about 3× (≈ 40–45% of the mass) on both routes (+0.00002 US/India, +0.00004 France route). v9 is kept at full weight on the France route, because its value (French fillers) is invisible to US/India validation.
- **First ensemble, E2h** (`src/ensemble.py`; superseded by the final 4-model ensemble above):
  - The two stacks' probabilities (mmBERT-small and Qwen) are averaged. Their out-of-fold correlation is 0.9967.
  - The decision rule is re-tuned on the averaged out-of-fold probabilities.
  - Stacked F_shift on train countries: mmBERT-small 0.99130, Qwen 0.99152, **average 0.99167**.
  - Unseen countries use the **average of the two transfer models** (`TRANSFER=mean`, F_shift 0.99117 vs 0.99074 for mmBERT-small alone and 0.99100 for Qwen alone). Qwen's France changes alone did not pass our audit (see §5); averaging keeps its correct rejections and dampens its errors.
- **Tried and not used:**
  - A CatBoost secondary: the log-loss-optimal blend weight was 0.
  - Isotonic calibration: not chosen on the final models.
  - `mmBERT-base` as the cross-encoder with self-training on unseen countries: public leaderboard 0.98791 (< v4); it accepted more sibling pairs in France.

### 4.4 Decision rule
- Per-S1 **expected-F0.5 maximisation**: an exact Poisson-binomial computation over the top-k candidates, with the empty set scoring 1 for singletons.
- A logit offset δ is chosen on F_shift for each route. In the final ensemble the best value is δ = 0 on both routes (v4: 0.25 full, 0.30 transfer).
- F_shift varies by < 0.0003 over δ ∈ [0, 1] (flat), so the decision is robust.

### 4.5 Unseen-country guards (`write_outputs.py`)
Each guard is re-derived from train labels at write time. None uses test labels. Pairs they flag are not matched.

| Guard | Pattern | Train truth | Applies to | France pairs removed (final) |
|---|---|---|---|---|
| Concept conflict | sibling: real-word swap at the same house number | US 0.77, India 0.08: they conflict, and both are below the F0.5 break-even of 0.8 | unseen countries | 6,595 |
| Hard negative | same name, **disjoint** legal forms (SARL → SASU; not a subset such as "Private Limited" → "Ltd"), house number shifted by 1–50, or no address | below 0.1 in every train country (India 0.000, US 0.009; the few true ones are PC/LP professional suffixes) | unseen countries | 23 |
| **Hard positive** (adds matches) | record name = S1 name with its category word replaced or extended by the country's mined **filler** words only ("Nantes Sport SARL" → "Nantes SARL Développement"), same house number and street | **≥ 0.95 in every train country** (US 0.995 of 246k, India 0.978 of 29k); in France the cross-encoders had never seen French fillers and accepted e.g. only 76% of "Développement" pairs vs 97–99% of other fillers | unseen countries, env `ER_HARD_POSITIVE=1` | **4,053 added** |
| Country word | record name = S1 name + the S1 country's own name ("EHPAD Joseph" → "EHPAD JOSEPH (FRANCE)") | India 1/452 true; the reverse case, where the S1 carries the country name and the record drops it, is 79–92% true and is not touched | all countries | 100 |

- **Why they are needed.** On train, the models accept these patterns at their true rate. The transfer model accepted them in France 8–30 times more often: the sibling pattern 11×, legal-swap near-misses 6.6% vs ≤ 0.8%.
- **How they were found.** By reading France decisions qualitatively, then confirming each pattern on train labels.
- **Scope.** A systematic scan of about 100 discrete patterns (name, number, street, legal-form and vocabulary relations) found no other near-zero-truth pattern that France accepts more often than train.

---

## 5. Results & Error Analysis

| Version | Candidates per test S1 | F_val | F_shift | **Public LB** |
|---|---|---|---|---|
| v1: stage 2 + cross-encoder stacking, 16-candidate rule | 7.88 | 0.99184 | 0.99162 | **0.98030** |
| v4: top-2 candidates + filler, real-word and acronym features + country routing | **4.89** | 0.99158 (full) / 0.99109 (transfer) | 0.99130 / 0.99074 | **0.98830** |
| v5: mmBERT-base cross-encoder + self-training on unseen countries | 4.89 | | 0.99144 / – | 0.98791 |
| E2h: v4 + Qwen3-1.7B ensemble (all countries, `TRANSFER=mean`) + unseen-country guards | **4.89** | | **0.99167** / 0.99117 | **0.990013** |
| E4: E2h + French house-number parsing fix (test features recomputed) | **4.89** | | 0.99167 / 0.99117 (train unchanged) | |
| E5a: E4, France route averaged with a Qwen3-1.7B retrained with French-format augmentation | **4.89** | | 0.99167 / 0.99110 | **0.990078** |
| E5r: E5a + hard-positive filler rule for unseen countries | **4.89** | | 0.99167 / 0.99110 (train unchanged) | |
| E6r: E5a with France's Qwen replaced by the filler-augmented continuation (v9) + hard-positive filler rule | **4.89** | | 0.99167 / 0.99110 |  |
| G3: US/India = mmBERT-small + Qwen3-4B; France = mmBERT-small + Qwen3-4B + v9 (3-way); + hard-positive filler rule | **4.89** | | 0.99171 / 0.99121 | **0.990543** |
| G4: both routes = mmBERT-small + Qwen3-1.7B + Qwen3-4B + v9 (4-way, equal weights); + hard-positive filler rule | **4.89** | | 0.99176 / 0.99126 |  |
| **final: as G4 with validated weights: US/India 1.5 / 1.5 / 3 / 0.75, France 1.5 / 2 / 3 / 1 (mmBERT-small / Qwen3-1.7B / Qwen3-4B / v9)** | **4.89** | | 0.99179 / 0.99127 | **0.990654** |

- F_shift barely changed between v1 and v4, while the public leaderboard rose by 0.0080.
- On test, v4 changed 23% of France's S1 decisions and only 2–6% in US and India. The gain came from the unseen country, as the diagnosis predicted.
- Label-free test diagnostics for v1:
  - uncertain pairs per S1: US 0.26, India 0.45, **France 0.95** (validation 0.23–0.28);
  - expected F0.5 per S1, **lowest for France**;
  - SHAP: unseen legal-form-pair categories and French names moved France near-misses upward.

**Validation errors (v4, full model):** 2,201 false-positive pairs and 38,378 false-negative pairs over 441,235 S1s.

- **Common false positives:**
  - residual synthetic near-miss distractors: 741;
  - records without an address, matched on name and sibling consensus: 531;
  - co-located businesses with different names: 322;
  - same-name look-alikes with a different number: 163.
- **Common false negatives:**
  - **address missing: 30,598 (80%).** Name-only records whose names are shared by several branches of a chain. Under F0.5 the model correctly declines to guess.
  - heavy name rewrites with an equal number: 2,381;
  - same name with digit-drop noise in the number: 1,281.

**Audit of every candidate submission against v4.** Reading examples only; no labels were produced or used.
- Every changed pair is put in a pattern bucket, and each bucket's train truth rate is shown next to it.
- Examples of changed France S1s are read with all their candidates.
- What the audits found:
  - **The Qwen transfer model alone** accepted more name-only and low-truth pairs in France. It also rejected clear matches whose only difference is a typo (for example "WU Sportive" for "WO Sportive").
  - Qwen does reject what v4 wrongly accepts: legal-swap near-misses ("MX Atelier SASU, 37 Rue Labottière" → "MX Atelier SNC, 39") and country-word distractors. These became the hard-negative and country-word guards, applied to the robust model.
  - The final submission changes 6.4% of France S1s vs v4, about half of them guard removals. US and India change on 0.6% of S1s: the ensemble decisions.

**Qualitative audit of France test decisions.**
- Handled correctly: typos, legal variants, domains, aliases, acronyms, department vs region, filler variants and invented brand names.
- Near-misses and sibling businesses are rejected, as learned on train.
- Remaining inconsistencies:
  - a few sibling businesses are still matched;
  - a few filler or brand variants are still missed.

---

## 6. Conclusion

**What worked, by measured impact:**
1. **Diagnosing the unseen country without labels, then generalising to it.** Leave-one-country-out training as the evaluator, country routing to transfer-robust models, and language-independent vocabulary features mined from each country's own data (fillers and real words) raised the public leaderboard from 0.98030 to **0.98830**.
2. **Test-like validation.** Synthetic near-misses at test rates exposed a generator bug and set the decision rule.
3. **A small, learned candidate set.** 4.89 candidates per S1 with no F0.5 loss; retraining on short lists *improved* the matcher.
4. **The listwise cross-encoder with a NIL option, stacked with sibling consensus.** +0.0027 to +0.0040 F_val over stage 2 alone.
5. **Per-entity expected-F0.5 decisions.**

**What did not help:** a CatBoost blend (weight 0), isotonic calibration on the final models, char-n-gram blocking, and dropping stage-1 score features for transfer.

**Open-set recognition vs domain shift.**
- The open-set components (EVT, EVM, NIL option, open-set losses) handle *unknown identities*: distractors. Together with the later features, validation false positives fell from 10,004 (first model) to 2,201.
- The unseen country is a *covariate shift*, which score-based open-set methods cannot fix, because they are calibrated on the training domain.
- Closing it needed domain-generalisation tools: LOCO evaluation, transfer-robust features and vocabulary mined per country.

**Label-free guards for the unseen country.** Patterns that are pure distractors in every train country (country-name additions; legal-form swaps with shifted numbers), or whose truth conflicts between train countries (sibling swaps), are not matched in France. Each is justified by train truth rates alone.

**Remaining error:** about 80% of false negatives are name-only records of chains, which no field can disambiguate. In France, a small number of sibling-vs-filler word swaps remain ambiguous without French knowledge.

---

## Appendix

### A. Code Artefacts
The code is in `src/` (run scripts in `scripts/`). `scripts/run_all.sh` runs every stage in order and can resume from any stage:
`prep → mine → normalize → synth → dense → block → stage1train → stage1test → evm → train → train_tr → write → xenctrain → xenc → trainx → trainx_tr → writex`.
`scripts/run_final.sh` then trains the other cross-encoder runs (Qwen3-1.7B; Qwen3-1.7B + format augmentation; its filler-augmented continuation v9; Qwen3-4B) on run A's candidates and stage-2 models, builds each run's stacked models, and writes the final `output/` with the weighted 4-run ensemble (`src/ensemble.py`) and the guards. With the saved runs, `START_FROM=ensemble ./scripts/run_final.sh` reproduces the submitted files byte for byte. `scripts/run_ensemble.sh` builds the earlier two-run ensembles (E1h/E2h/E5a). The exact Qwen3-4B training code and logs (2 folds × 352,000 records, 11,000 steps each, 2 GPUs per fold) are in `experiments/qwen4b_remote_training/`. The official validator runs after each write. Measurement scripts are in `experiments/` (candidate size, transfer/LOCO, LLM throughput).

### B. Compliance
- **No external data, APIs or lookups.**
- **Mined vocabularies.** Every dictionary and vocabulary is mined from the provided data, either from labelled train pairs or from unlabelled test data (TF-IDF vocabulary, address pseudo-pairs, filler and real-word vocabularies for countries absent from train). Hand-written knowledge is limited to generic regexes and a short list of legal-form and honorific seed words.
- **Country handling.** Country is an open set of labels: every per-country structure is built for whatever countries appear in the data. Routing depends only on whether a country appears in train. The guards and the filler rule are re-derived from train labels at write time and apply to any country absent from train.
  - **Exception (disclosed):** the cross-encoder *training augmentation* (`cross_encoder._augment`: `AUG_DST_LEGAL`, `AUG_COUNTRY`, `AUG_NUM_FMT`, `AUG_FILL_DST`) contains hand-written French surface forms: legal forms (SARL, SAS, SASU, EURL, …), the word "France", house-number prefixes (`N°`, `(12)`), and fillers (Groupe, Développement, & Associés, Et Fils, Services).
  - They were chosen from **unlabelled** test records (the fillers from our label-free mined list `fillers.json`) and general knowledge of French business names.
  - They only rewrite *train* examples, so labels are preserved; no test label is used.
  - The validation re-weighting refers to the documented test mix.
- **Models.** All pretrained weights are MIT or Apache-2.0 and ≤ 8B parameters: `multilingual-e5-small` (MIT), `mmBERT-small` (MIT, 0.14B), `Qwen/Qwen3-1.7B` (Apache-2.0, 1.7B, LoRA) and `Qwen/Qwen3-4B` (Apache-2.0, 4B, LoRA). Revisions are pinned. Libraries are permissive: XGBoost and CatBoost Apache-2.0, libMR BSD-3. See `LICENSES.md`.
- **Test data use.** Unlabelled test data is used only transductively: vocabularies, the augmentation surface forms above, and the label-free audit of decisions. It is never used with labels. Test predictions were inspected only qualitatively for pipeline bugs and systematic patterns; every rule we derived from them was checked against *train* labels before use. No test labels were created or used. The public leaderboard was used only to choose between finished submissions.
