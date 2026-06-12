# alexmueller07.github.io

This is not a website. It is a small neural network trained on exactly one person.

Live at [alexmueller07.github.io](https://alexmueller07.github.io).

There are no pages and nothing to scroll. The whole screen is an explorable network:

- Drag to pan, scroll to zoom
- Click a labeled output neuron, or type a question in the prompt bar
- Watch the forward pass ripple through the layers
- The answer streams out of the decoder token by token

Out-of-distribution questions get the uncertainty they deserve.

Vanilla HTML, CSS, and JavaScript. No frameworks, no build step.

## Run locally

```
python -m http.server 8000
```

Then open http://localhost:8000.
