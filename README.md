# alexmueller07.github.io

This is not a website. It is a real neural network trained on exactly one person.

Live at [alexmueller07.github.io](https://alexmueller07.github.io).

## How it works

A 2,979-parameter MLP (128 hashed char-trigram features → 20 relu → 14 relu → 7 softmax)
is trained in Python on a few hundred phrasings ([train/train.py](train/train.py)),
its weights exported to [js/weights.json](js/weights.json), and executed in the browser
with hand-written matrix math. When you type a question:

1. Your words are hashed into 128 features
2. A real forward pass runs through two hidden layers
3. Every neuron on screen glows with its true activation
4. The brightest output neuron wins, and the answer streams out of it token by token
5. The `p=` value is the actual softmax confidence

Questions the model wasn't trained on get routed to the `???` neuron,
where they receive the uncertainty they deserve.

There are no pages and nothing to scroll. Drag to pan, scroll to zoom,
click neurons, or ask the prompt bar anything.

Vanilla HTML, CSS, and JavaScript. No frameworks, no build step.

## Retrain it

```
python train/train.py    # writes js/weights.json
python -m http.server 8000
```
