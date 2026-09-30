# ami-trim-5

Lab submissions for the MCA trimester 5 course at Christ University.

## Lab 1: Q-learning with an epsilon-greedy policy

- `Q_Learning_Epsilon_Greedy.ipynb` - the lab notebook. Trains a tabular Q-learning agent on Gymnasium's `Taxi-v4` (500 discrete states, 6 discrete actions, deterministic transitions, non-continuous) across five training budgets (500, 1000, 2000, 5000, 10000 episodes), with an epsilon-greedy policy that starts fully exploratory and decays toward a small exploration floor. Records per-episode reward, plots cumulative reward and a 100-episode moving average, tests each learned policy for 100 greedy episodes, and visualises the environment and the extracted policy.
- `Q_Learning_Epsilon_Greedy.pdf` - a rendered PDF export of the executed notebook.

## Lab 2: CNN feature extraction and LeNet-5 classification (`lab-2/`)

- `CNN_Latent_Features.ipynb` - the lab notebook, in TensorFlow/Keras.
  - Part A1: a PneumoniaMNIST chest X-ray and a BreastMNIST breast ultrasound (both grayscale) go through the same small CNN. Conv1 holds 8 hand-written kernels from [setosa.io/ev/image-kernels](https://setosa.io/ev/image-kernels/) (blur, sharpen, emboss, outline, four Sobels). Conv2 and Conv3 are built-in `Conv2D` layers with 16 and 32 filters, and a Dense layer gives a 64-D latent vector. The two modalities are then compared filter by filter and by latent-vector similarity.
  - Part A2: an RGB cat photo (`skimage.data.chelsea`) goes through 8 hand-written filters (Sobel edges, Gabor textures, blob shapes) and through pretrained VGG16, whose global-average-pooled output is a 512-D latent vector.
  - Part B: LeNet-5 on MNIST (padded to 32x32, 55k/5k/10k train/val/test). Seven hyperparameters are swept one at a time from a baseline, each with 3 seeds and 10 epochs: learning rate, batch size, activation, pooling, optimizer, dropout, and learned vs user-defined (frozen) C1 filters. The best model scores 98.99% on the test set, and its 84-D F6 latent is compared with raw pixels.
  - Complexity element: convolution, ReLU and max pooling are written by hand in NumPy and checked against TensorFlow's built-ins at every step, including VGG16's first-layer weights and the trained LeNet-5's C1.
  - Shows every filter, the feature maps after each conv and pool layer, all three architectures, the flattened features, the latent vectors, and a side-by-side view of edge, texture, shape, part and semantic features.
- `CNN_Latent_Features.pdf` - a PDF export of the executed notebook.
- `Pixel to Latent Representation.ipynb` - the base code the lab builds on.
- `outputs/` - the three latent vectors (X-ray, ultrasound, cat) as CSV.
- The PneumoniaMNIST (214 MB) and BreastMNIST (31 MB) files are not in the repo, and the notebook downloads them into `lab-2/data/` on first run (MNIST comes through Keras). A full run takes about 20 minutes, most of it the LeNet-5 sweep.

## Lab 3: Text to a flattened latent vector (`lab-3/`)

- `Text_Latent_Representation.ipynb` - the lab notebook, in TensorFlow/Keras.
  - Part A covers the 8 tasks on five customer reviews: cleaning, a 16-word vocabulary (18 IDs with `<PAD>` = 0 and `<OOV>` = 1), token sequences, post-padding to 7, an 18 x 8 embedding trained on sentiment, the 5 x 7 x 8 embedding tensor, one 56-D flattened vector per review, and 2-D PCA of the words and the reviews.
  - Every stage is written by hand and checked against the Keras built-in (`TextVectorization`, `pad_sequences`, `Embedding`, `Flatten`), and each step ends with an `assert`.
  - Part B (self-learning): the findings re-checked over 5 seeds, Flatten vs average pooling (average pooling puts 99.9% of the learned change on one axis), a NumPy skip-gram word2vec with gradient checking (it puts "love" next to "dislike" with no labels), and unseen reviews with out-of-vocabulary words and negation.
  - Ends with a checklist that maps each task to the expected output it verified.
- `Text_Latent_Representation.pdf` - a PDF export of the executed notebook.
- `outputs/flattened_vectors.csv` - the five 56-D flattened review vectors.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt           # Lab 1
pip install -r lab-2/requirements.txt     # Lab 2
pip install -r lab-3/requirements.txt     # Lab 3
jupyter notebook
```
