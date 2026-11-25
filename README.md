# alexmueller07.github.io

My personal site. Live at [alexmueller07.github.io](https://alexmueller07.github.io).

One page of plain HTML and CSS. No framework, no build step, no JavaScript.
The layout and look are adapted from [Will Huey's homepage](https://willhuey.com).

```
index.html          the page
assets/site.css     styles
assets/             tab icon, photo, CV and paper figures
nn/                 the previous version of this site
```

To preview locally:

```
python -m http.server 8000
```

## Updating

- **Photo:** the header uses `assets/alexander-mueller.jpg`. Replace that file to change it.
- **Tab icon:** `favicon.ico` and the `favicon-*.png` / `apple-touch-icon.png` files in `assets/`, made from a drawing of me.
- **CV:** save the PDF as `assets/cv.pdf`. The CV link in the header already points there.
- **Paper links:** once a preprint is public, add `<p class="pub-links"><a href="...">arXiv</a></p>` under that paper's summary in `index.html`. The pill style is already in `site.css`.
- **News line and "Last updated":** both are plain text in `index.html`.

## The previous site (`/nn`)

The old site was not a website. It is a real neural network trained on exactly one person,
and it still runs at [alexmueller07.github.io/nn](https://alexmueller07.github.io/nn/).

A 2,979-parameter MLP (128 hashed char-trigram features → 20 relu → 14 relu → 7 softmax)
is trained in Python on a few hundred phrasings ([nn/train/train.py](nn/train/train.py)),
its weights exported to [nn/js/weights.json](nn/js/weights.json), and executed in the browser
with hand-written matrix math. When you type a question:

1. Your words are hashed into 128 features
2. A real forward pass runs through two hidden layers
3. Every neuron on screen glows with its true activation
4. The brightest output neuron wins, and the answer streams out of it token by token
5. The `p=` value is the actual softmax confidence

To retrain it:

```
cd nn
python train/train.py    # writes js/weights.json
```
