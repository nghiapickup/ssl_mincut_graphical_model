Simple graphical model for graph-based semi-supervised learning
===

This project derives a simple graphical model for the objective function of the 
[mincut approach][1].

For more detail your may refer to chapter 3 of [my master thesis][2].

Note: Inference algorithms are mostly implemented as they are.

How to reproduce
---
- Install requirements.txt
- All test cases are implemented in `source.py` as individual functions.

Main modules
---
- `data/...` defines data processing scripts
- `graph_construction` constructs graph from input data, 
supports some simple graph construction methods
- `?_inference` defines graph-based model inference.

[1]: http://www.aladdin.cs.cmu.edu/papers/pdfs/y2001/mincut.pdf
[2]: https://github.com/nghiapickup/master_thesis/blob/master/thesis.pdf