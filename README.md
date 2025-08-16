# ML-Practical-Assignments

A collection of 8 hands-on mini-projects spanning core Machine Learning topics—from classic algorithms built from scratch to ensemble methods and practical evaluation. Each project lives in its own directory with a Jupyter notebook that you can run end-to-end.

## Quick start

```bash
# Python 3.9+ recommended
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -U pip
pip install jupyter numpy pandas scikit-learn matplotlib seaborn
# Some projects may also use: pillow, graphviz (plus system Graphviz), imageio, ipywidgets
```

Launch notebooks:

```bash
jupyter lab   # or: jupyter notebook
```

> Tip: All notebooks set seeds (e.g., `random_state=41` or `42`) for reproducibility where relevant. Results still may vary slightly across platforms and library versions.

---

## Project 1 — Perceptron from Scratch

**Path:** `HW1-2 Perceptron/Perceptron.ipynb`
**Goal:** Implement and train a binary Perceptron classifier on a linearly separable dataset; visualize the learned decision boundary in 2D.

### What’s inside

* Synthetic data via `sklearn.datasets.make_blobs`:

  * 100 samples, 2 or 10 features, 2 centers, `random_state=41`.
  * Labels mapped to `{-1, +1}`.
* Minimal components:

  * `F.sign(x)`: static sign activation.
  * `Perceptron`: weights initialized to ones, bias to 0; forward pass `sign(w·x + b)`.
  * `Optimizer.update`: Perceptron learning rule
    `if y != y_hat: w ← w + y·x,  b ← b + y`.
* Training loop:

  * Up to 1000 iterations over the dataset.
* 2D visualization (when `n_features=2`):

  * Scatter plot of both classes.
  * Decision line: `x2 = -(w0/w1) * x1 - b/w1`.
  * Prints final weights and bias.

### Run it

Open the notebook and run all cells. To reproduce the 2D plot section, ensure `n_features=2` in the dedicated cell.

### Expected outputs

* A plot showing two separable clusters and the learned linear separator.
* Final `weights` and `bias` printed to console.

### Notes and possible extensions

* Add a learning rate and shuffle samples per epoch to speed convergence.
* Introduce an early-stopping criterion (stop when no updates occur).
* Track accuracy over epochs; separate a validation split.
* Try different initializations and margin variants (e.g., Perceptron with margin).

---

## Project 2 — KNN & Ensembles for Employee Attrition

**Path:** `HW2-1 KNN Ensemble/KNN_Ensemble.ipynb`
**Goal:** Predict employee attrition using EDA, preprocessing, a KNN classifier implemented from scratch, and ensemble baselines (Random Forest, Bagging, AdaBoost).

### Data

* `dataset.csv` (train): includes features and target `Attrition` (with some “masked” values used for automated tests).
* `test.csv` (eval): encoded features only.
* The notebook writes predictions to `result.csv` with one column: `target`.

### Workflow

1. **EDA**

   * Inspect schema (`.info()`, `.describe()`), missing values, and target distribution.
   * Drop columns with fewer than 2 unique values.
   * Visuals:

     * Average `MonthlyIncome` vs `YearsAtCompany` (line plot).
     * Department-level average income; prints highest/lowest departments.

2. **Preprocessing**

   * Label-encode all categorical columns.
   * Train/test split: 70/30 (`random_state=42`).
   * Standardize features with `StandardScaler`.

3. **KNN from scratch**

   * `CustomKNN(k)` with Euclidean distance.
   * Search `k ∈ {1,…,17}`; pick the best by test accuracy.
   * Report: accuracy, classification report, confusion matrix heatmap.

4. **Evaluation on held-out file**

   * Load `test.csv`, scale with the **already-fit** scaler, predict with the best KNN, and save `result.csv`.
   * Important: in a production setting, call `scaler.transform(...)` on eval data (do **not** re-fit the scaler on `test.csv` to avoid leakage).

5. **Ensemble baselines**

   * **Random Forest** with `GridSearchCV`
     Grid: `n_estimators ∈ {100,200,300}`, `max_depth ∈ {10,20,None}`, `min_samples_split ∈ {2,5,10}`.
     Reports accuracy, classification report, confusion matrix, and feature-importance bar plot.
   * **Bagging with KNN**: `BaggingClassifier(KNeighborsClassifier(n_neighbors=bestk), n_estimators=50)`.
   * **AdaBoost (SAMME)** with grid over `n_estimators ∈ {50,100,150}`, `learning_rate ∈ {0.01,0.1,1.0}`.

6. **Model comparison**

   * Line plot comparing accuracies of: KNN (custom), Random Forest, Bagging KNN, AdaBoost.

### Run it

Ensure the CSV files are in the same directory as the notebook, then run all cells. The notebook will:

* Produce EDA plots and metrics,
* Train/evaluate all models,
* Write `result.csv` for the external evaluation set.

### Notes and tips

* Always standardize features for distance-based models like KNN.
* Consider stratified splits if the classes are imbalanced.
* For cleaner evaluation, use cross-validation instead of a single hold-out split.
* Double-check that scalers/encoders are **fit on the training set only** and reused via `.transform(...)` for test/eval sets.

---

## Project 3 — Optimization: NumPy MLP + Optimizer Experiments

**Path:** `HW3-2 Optimization/P2_Questions.ipynb`
**Goal:** Build a 1-hidden-layer neural net **from scratch** (NumPy) and study optimization: adaptive methods (Adam/RMSprop), classic SGD, momentum, learning-rate effects, and mini-batch vs. full-batch—using (Fashion-)MNIST.

### What’s inside

* **Data & prep**

  * Loads `tf.keras.datasets.fashion_mnist` (28×28 grayscale), normalizes to `[0,1]`, one-hot encodes labels.
  * Flattens images to vectors for a tiny MLP.
* **Model (manual NumPy implementation)**

  * Parameters: `W1,b1,W2,b2`; hidden **ReLU**; output activation implemented with **sigmoid**; cross-entropy loss.
  * Forward pass, backprop, and plain GD updates coded step-by-step.
* **Experiments**

  * Vary hidden size (`[32, 64, 128]`) and observe training loss.
  * **Adam**: implements moment estimates (`v,s`) + bias correction; training loop with Adam.
  * **RMSprop**: moving average of squared grads; compares **Adam vs RMSprop** (loss curves).
  * **SGD**: effect of learning rates (`[1e-3, 1e-2, 1e-1]`) on convergence.
  * **SGD vs. SGD+Momentum**: velocity accumulator, comparison plots.
  * **Mini-batch vs Full-batch**: configurable batch size, tracks epoch loss/accuracy, simple timing.
* **Plots**

  * Loss curves for optimizer comparisons; printed snapshots (epochs 0/10/20/30/40) for quick inspection.

### Run it

```bash
pip install numpy matplotlib tensorflow
jupyter lab  # open the notebook and run all cells
```

### Expected outputs

* Console logs with losses per experiment.
* Matplotlib figures comparing:

  * Adam vs RMSprop loss,
  * SGD learning-rate sweeps,
  * SGD vs Momentum,
  * Mini-batch vs Full-batch (loss/accuracy over epochs).

### Notes & tips

* The notebook uses a minimal didactic MLP; don’t expect state-of-the-art accuracy.
* If training is slow on CPU, reduce `epochs` or sample a subset of data for the demos.
* Ensure array shapes line up: inputs are `(m, 784)`, `W1` is `(hidden, 784)`, etc.
* For fair SGD comparisons, keep the same initialization across runs (set a NumPy seed).

---

## Project 4 — Optimizers from Scratch + Keras-Tuned MNIST

**Path:** `HW3-3 Optimizers/P3_Questions_Extra (1).ipynb`
**Goal:** Implement five optimizers (SGD, Momentum, Adagrad, RMSprop, Adam) in NumPy and visualize their behavior on a **non-convex 1D function**; then use **Keras Tuner** to build a simple dense network for MNIST and target **≥98%** test accuracy. Save the trained model and visualize results.

### What’s inside

* **Optimizers (NumPy, from scratch)**

  * `SGDClass`, `SGDWithMomentumClass`, `AdagradClass`, `RMSpropClass`, `AdamClass`; each tested on toy weights/gradients.
* **Non-convex objective demo**

  * Function: $f(x)=4\cos(x-1) + \frac{\sin(2\pi x)}{x}$ with analytic gradient.
  * Start at `x=1.9`, run each optimizer for 400 iterations; record `x` and `f(x)`; plot trajectories and per-iteration loss.
  * Discussion prompts on **local minima**, **saddle points**, **plateaus**, **oscillation**, and **convergence**.
* **MNIST model with hyperparameter tuning**

  * Data: `tf.keras.datasets.mnist` (28×28), normalized + one-hot.
  * Model: `Flatten → Dense(units, ReLU) → Dropout → Dense(10, softmax)`.
  * Tuning via **Keras Tuner (Hyperband)**:

    * `units ∈ {64,128,192,256}`, `dropout ∈ {0.0,0.2,0.3,0.4}`, optimizer ∈ {adam, rmsprop, sgd}.
  * Train best trial, evaluate on test set; saves as `mnist_model.keras`.
  * **Visualizations**: training/validation accuracy curves, confusion matrix heatmap, and a 4×5 grid of misclassified digits.

### Run it

```bash
pip install numpy matplotlib seaborn tensorflow keras-tuner
jupyter lab  # open and run the notebook
```

> If your environment doesn’t like spaces/parentheses in file paths, consider renaming the notebook file or open it directly from Jupyter’s browser UI.

### Expected outputs

* **Optimizer demo**: individual trajectory plots overlaid on the loss curve + combined convergence plot.
* **Tuning report**: best hyperparameters (units/dropout/optimizer), training logs, and **test accuracy** printout (aiming for ≥0.98).
* **Saved model**: `mnist_model.keras`.
* **Diagnostics**: confusion matrix heatmap and misclassified examples.

### Notes & tips

* For reproducible tuning, set seeds for NumPy/TensorFlow before calling the tuner.
* GPU is optional; this model trains quickly on CPU. Increase `epochs` if accuracy stalls.
* Keras Tuner writes artifacts under `hyperband_dir/`; delete it to restart a clean search.
* The non-convex demo uses scalar updates—your optimizer classes handle NumPy arrays and will work with scalars via broadcasting.

---

## Project 5 — CNN: MobileNet V1 & V2 (PyTorch), Transfer Learning & Distillation

**Path:** `HW4-1 CNN MobileNet V1 V2/P1_Questions (5).ipynb`
**Extra artifact:** `HW4-1 CNN MobileNet V1 V2/model.pt` (saved MobileNet-V1 weights)

### What’s inside

* **Datasets & transforms**

  * CIFAR-10 (and CIFAR-100 for TL).
  * Train transforms: `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `Normalize`.
  * Test/Val transforms: `Resize(224)`, `Normalize`.
* **Utilities**

  * `imshow` (denormalize & show), `fit_epoch`, `train`, `get_acc`, `plot_losses`, `count_parameters`.
* **MobileNet V1 (from scratch)**

  * `conv_bn` (3×3 conv → BN → ReLU), `conv_dw` (depthwise 3×3 + pointwise 1×1) blocks.
  * `MobileNet(n_class)` with stacked depthwise-separable convs + `AdaptiveAvgPool2d(1)` + FC.
  * Train on CIFAR-10 (default `epochs=10`, `lr=1e-3`), target **≥65%** val accuracy; logs training time.
  * **Speed/Params comparison** vs. a same-skeleton **Normal CNN** (standard 3×3 convs).
* **Transfer Learning to CIFAR-100**

  * Load `model.pt` weights into a `MobileNet(n_class=100)` (skip non-matching keys).
  * **Freeze** first N depthwise blocks (e.g., 7) and fine-tune final layers on CIFAR-100 (`epochs≈7`, `lr=2e-4`).
* **MobileNet V2 (from scratch)**

  * `InvertedResidual` with **expand → depthwise → linear 1×1** (+skip if stride=1 and in=out), **ReLU6** activations.
  * `MobileNetV2(n_class=10, width_multiplier=α)`; train on CIFAR-10; then sweep α ∈ {0.1,…,1.0} and print params.
  * Load a **pretrained α=0.5** model (`MNv2WMmodel.pt`) and compare val accuracy/time to α=1.0.
* **Knowledge Distillation**

  * Teacher: `timm` **ResNet-18** (pretrained).
  * Student: `MobileNetV2`.
  * `DistillationLoss` = α·KL(softmax(T)) + (1−α)·CE(hard), with temperature τ (default α=0.5, τ=3).
  * Quick sanity run (1 epoch) to verify training loop.

### Why it matters

* Depthwise separable convs cut compute & params by \~**8–9×** vs standard 3×3 (≈1/9 FLOPs + pointwise overhead).
* V2’s **linear bottlenecks** preserve information in the compressed space; **inverted residuals** ease optimization.
* **Width multiplier α** scales channels → ≈**α²** parameter scaling (classifier & BN add small offsets).
* **Resolution multiplier** speeds compute (fewer spatial ops) but doesn’t change parameter count.
* **KD** transfers “dark knowledge” for better small-model accuracy.

### Run it

```bash
pip install torch torchvision timm tqdm
# (Optional) ensure a GPU is visible: torch.cuda.is_available()
# Launch and run the notebook cells
```

### Expected outputs

* Training/validation loss curves; CIFAR-10 acc for V1/V2; timing printouts for V1 vs Normal CNN.
* Param counts for both models and for V2 with different α.
* CIFAR-100 fine-tune logs; KD epoch logs and student val accuracy.

---

## Project 6 — Skip-gram Word2Vec with Negative Sampling (TensorFlow)

**Path:** `HW5-1 Skipgram/ml-hw5-p-q1.ipynb`
**Artifacts:**

* `HW5-1 Skipgram/word2vec_model_checkpoint.weights.h5` (trained weights)
* `HW5-1 Skipgram/meta.tsv` & `vecs.tsv` (embeddings + labels for the Embedding Projector)

### What’s inside

* **Data**

  * **text8** (via `gensim.downloader`), using \~15% subset for faster runs.
* **Preprocessing**

  * Lowercase, strip punctuation, **stopword removal** (NLTK), **min-freq ≥5** filter.
  * **Subsampling** frequent words with Mikolov heuristic (t=1e-5).
  * Tokenization with Keras `Tokenizer`.
* **Pairs & negative sampling**

  * Build (target, context) with `keras.preprocessing.sequence.skipgrams` (`window=5`, `negative_samples=0.5`).
  * Split into train/test; wrap in `tf.data` pipelines with shuffle/batch.
* **Model (subclassed `tf.keras.Model`)**

  * `Embedding(vocab, d)` + dot-product between target/context embeddings → small `Dense(1)` head.
  * Custom **train/test step** with `tf.GradientTape`, **binary cross-entropy**, **Adam** optimizer, accuracy metrics.
* **Saving & visualization**

  * Save weights to `word2vec_model_checkpoint.weights.h5`.
  * Export embeddings to **`vecs.tsv`** and labels to **`meta.tsv`** for TF **Embedding Projector**.

### Run it

```bash
pip install tensorflow gensim nltk tensorflow-datasets
python -c "import nltk; nltk.download('stopwords')"
# Launch and run the notebook cells end-to-end
```

### Expected outputs

* Epoch logs: training/validation **binary accuracy** and cumulative losses; epoch timings.
* Learned embedding matrix shape `(vocab_size, 128)`.
* `vecs.tsv` & `meta.tsv` ready to upload at [http://projector.tensorflow.org](http://projector.tensorflow.org) for interactive exploration.

### Tips

* If memory is tight, reduce corpus fraction or `BUFFER_SIZE`/`BATCH_SIZE`.
* For clearer neighborhoods in the projector, search words like countries, animals, or professions.
* Ensure the loss/activation combo is consistent (logits vs sigmoid) if you customize the head.

---

## Project 7 — HW5-2 NLP Transformer & BERT (two notebooks)

### A) `ml_hw5_p2_s1_Transformer.ipynb` — Transformer from scratch (PyTorch)

**What I built**

* Full Transformer stack coded by hand:

  * `InputEmbeddings`, `PositionalEncoding`, `LayerNormalization`, `FeedForwardBlock`
  * `MultiHeadAttentionBlock` (scaled dot-product + masking + multihead split/concat)
  * Residual + Norm wrapper, `EncoderBlock`/`Encoder`, `DecoderBlock`/`Decoder`
  * `ProjectionLayer` (to vocab) and a `Transformer` wrapper (`encode`, `decode`, `project`)
* Training utilities:

  * Word-level tokenizer (HF `tokenizers`) with `[UNK] [PAD] [SOS] [EOS]`
  * Dataset: **OpusBooks** (English→Italian), `BilingualDataset` + causal mask (`casual_mask`)
  * Greedy decoding + validation printer
  * Training loop (Adam, CE with label smoothing), TensorBoard logging, checkpointing

**Key configs (paper-like defaults)**

* `d_model=512`, `N=6` layers, `h=8` heads, `d_ff=2048`, `dropout=0.1`, `seq_len=64`, `batch_size=32`.

**How to run**

```bash
pip install torch torchvision datasets tokenizers tqdm tensorboard
# run all cells; GPU strongly recommended
```

**Gotchas / tips**

* Make sure the tokenizer JSONs get saved/loaded to the paths in `config['tokenizer_file']`.
* Padding IDs: training uses target PAD for both masks; validation uses source PAD for the encoder mask. Keep PAD tokens consistent across tokenizers (or adapt the mask logic).
* To resume, set `config['preload']` to a saved `*.pth`.

---

### B) `ML_HW5_P_Q2_S2_Bert.ipynb` — BERT & TinyBERT+LoRA for complaint classification

**What I built**

* **Part 1 (data):** Load a *small* “Consumer Complaint” CSV from Google Drive ZIP → sample 15% → balance (keep major classes) → clean text → label encode.
* **Part 2 (BERT base):**

  * Tokenize with `bert-base-uncased` (max\_len=128), `ComplaintDataset`, train/test split.
  * `BertForSequenceClassification` + AdamW; 2–3 epochs; report loss/accuracy; save model+tokenizer to Drive.
* **Part 3 (LoRA on TinyBERT):**

  * Base model `prajjwal1/bert-tiny`, PEFT with LoRA (`r=8`, `alpha=16`, `dropout=0.1`, targets `["query","value"]`).
  * Train 2–3 epochs; evaluate; save LoRA adapter + tokenizer.

**How to run**

```bash
pip install transformers peft scikit-learn pandas tqdm
# (in Colab) mount Drive and ensure the ZIP path exists
```

**Notes**

* If you’re not in Colab/Drive, replace the Drive paths with local ones.
* Batch sizes: 16 (fits most GPUs). Increase epochs for stronger results.
* Parameter footprint: LoRA trains only small adapters → massive drop in trainable params vs full fine-tune.

---

## Project 8 — HW6 Contrastive Learning (CLIP-style KD)

**Notebook:** `HW6 Contrastive Learning/HW6_P_400105433.ipynb`

**Goal**

* Distill a large **text** encoder (teacher: OpenCLIP `EVA02-E-14-plus`) into a smaller multilingual **student** (`setu4993/smaller-LaBSE`) via **symmetric contrastive loss** with a learnable **temperature**.

**Pipeline**

* **Data:** Paired English–Persian CSVs (`train.csv`, `val.csv` via `gdown`).
* **Preprocess:** Language-specific normalization; compute token-length percentiles (`tok_percentile=99`) to cap `max_length` per tokenizer.
* **Models:**

  * **Teacher:** OpenCLIP `TextTransformer` (frozen).
  * **Student:** HF AutoModel + `LinearProjection` to teacher dim (1024). Residual MLP head with Swish, BN, Dropout, LayerNorm.
* **Loss (both directions):**

  * Normalize embeddings → logits = sims \* temperature
  * `loss = (CE(ref→cand) + CE(cand→ref)) / 2`
  * “Accuracy” is computed in both directions and summed (so raw values ∈ \[0,2]); later multiplied by 50 to show **%**.
* **Training:** AdamW over student + temperature; ReduceLROnPlateau on validation accuracy; GPU-only guard.

**How to run**

```bash
pip install open_clip_torch transformers datasets pandas numpy matplotlib tqdm gdown
# Run all cells on GPU; ensure train/val CSVs download successfully
```

**Tips / small things to watch**

* The projection output size (`project_to=1024`) must match teacher embedding dim (already set).
* BatchNorm usage on the CLS slice expects consistent tensor shape; if you refactor, keep it `(B,H)` before BN for clarity.
* If you hit OOM: lower `batch_size`, or freeze more student layers.

---

# Final Notes

* All notebooks are self-contained and ordered by folder. Run top-to-bottom on GPU.
* Results vary with hardware, seeds, and data sampling; the code logs configs/paths at runtime.
* Checkpoints and tokenizer files are saved beside each notebook (or in the specified Drive paths).

# Quick Repro Guide

1. Create a fresh Python 3.10+ env.
2. Install per-notebook requirements at the top of each notebook (or `pip install -r requirements.txt` if you’ve created one).
3. Use a CUDA-enabled GPU (T4/P100/V100 or similar). Reduce batch sizes if you hit OOM.
4. Run cells in order; watch the printed config and saved artifact paths.

# Troubleshooting

* **Out of memory (CUDA):** lower `batch_size`, `seq_len`, or width multipliers; clear workspace and restart.
* **Tokenizer/checkpoint not found:** verify paths printed by the notebook; ensure Drive is mounted (Colab) or adjust local paths.
* **HF download rate limits:** set `HF_HOME` to a writable cache; retry cell.

# Acknowledgements

* Datasets: CIFAR-10/100, Text8, OPUS Books, Consumer Complaints (subset).
* Libraries: PyTorch, TensorFlow, Hugging Face (transformers, tokenizers, datasets, peft), timm, open\_clip.
* Architectures: MobileNet V1/V2, Transformer, BERT/TinyBERT, CLIP.

# License & Usage

This repository is for academic use. Please cite the upstream papers and follow the licenses of datasets and libraries used.

# Contact

Sina Beyrami — Sina.Bey743@gmail.com
Questions/feedback are welcome.
