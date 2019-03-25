Graphical model for graph-based semi-supervised learning
===

This project derives a simple graphical model for the same objective function as the 
[mincut approach][1].

Note: Inference algorithms are mostly implemented as they are.

How to reproduce
---
- Install requirements.txt
- All test cases are implemented in `source.py` as individual functions.

Main modules
---
- `data/...` all data processing scripts
- `graph_construction` construct graph from input data, 
support some simple graph construction methods
- `?_inference` graph-based model inference.

[1]: http://www.aladdin.cs.cmu.edu/papers/pdfs/y2001/mincut.pdf