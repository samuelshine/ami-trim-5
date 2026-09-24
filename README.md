# ami-trim-5

Lab submissions for the MCA trimester 5 course at Christ University.

## Lab 1: Q-learning with an epsilon-greedy policy

- `Q_Learning_Epsilon_Greedy.ipynb` - the lab notebook. Trains a tabular Q-learning agent on Gymnasium's `Taxi-v4` (500 discrete states, 6 discrete actions, deterministic transitions, non-continuous) across five training budgets (500, 1000, 2000, 5000, 10000 episodes), with an epsilon-greedy policy that starts fully exploratory and decays toward a small exploration floor. Records per-episode reward, plots cumulative reward and a 100-episode moving average, tests each learned policy for 100 greedy episodes, and visualises the environment and the extracted policy.
- `Q_Learning_Epsilon_Greedy.pdf` - a rendered PDF export of the executed notebook.

## Lab 2: CNN feature extraction, from pixels to latent vectors (`lab-2/`)

- `CNN_Latent_Features.ipynb` - the lab notebook, in TensorFlow/Keras.
  - Part A: a PneumoniaMNIST chest X-ray and a BreastMNIST breast ultrasound (both grayscale) go through the same small CNN. Conv1 holds 8 hand-written kernels from [setosa.io/ev/image-kernels](https://setosa.io/ev/image-kernels/) (blur, sharpen, emboss, outline, four Sobels). Conv2 and Conv3 are built-in `Conv2D` layers with 16 and 32 filters, and a Dense layer gives a 64-D latent vector. The two modalities are then compared filter by filter and by latent-vector similarity.
  - Part B: an RGB cat photo (`skimage.data.chelsea`) goes through 8 hand-written filters (Sobel edges, Gabor textures, blob shapes) and through pretrained VGG16, whose global-average-pooled output is a 512-D latent vector.
  - Complexity element: convolution, ReLU and max pooling are written by hand in NumPy and checked against TensorFlow's built-ins at every step, including VGG16's own first-layer weights.
  - Shows every filter, the feature maps after each conv and pool layer, both architectures, the flattened features, the latent vectors, and a side-by-side view of edge, texture, shape, part and semantic features.
- `CNN_Latent_Features.pdf` - a PDF export of the executed notebook.
- `Pixel to Latent Representation.ipynb` - the base code the lab builds on.
- `outputs/` - the three latent vectors (X-ray, ultrasound, cat) as CSV.
- The PneumoniaMNIST (214 MB) and BreastMNIST (31 MB) files are not in the repo, and the notebook downloads them into `lab-2/data/` on first run.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt           # Lab 1
pip install -r lab-2/requirements.txt     # Lab 2
jupyter notebook
```
