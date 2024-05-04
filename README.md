# Simple Simultaneous Ensemble Genetic Programming (2SEGP)

Python code for 2SEGP, which was proposed in:

```
@inbook{virgolin2021genetic,
  author = {Virgolin, Marco},
  title = {Genetic Programming is Naturally Suited to Evolve Bagging Ensembles},
  year = {2021},
  isbn = {9781450383509},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3449639.3459278},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  pages = {830–839},
  numpages = {10}
}
```

Differently from classic evolution which returns a single model, 2SEGP returns an ensemble.
Nice aspects of 2SEGP are that:
- it is a very simple modification of classic GP
- it has only 1 additional hyper-parameter compared to classic GP (ensemble size) and it is rather robust to different choices
- it runs almost as fast as classic GP (see Big-O's in the paper)
- despite being simpler than other existing approaches, it typically performs equal or better than them
- main ideas are portable to other evolution-based algorithms (e.g., neural architecture search)


## Installation
Run `pip install .` from inside this folder.

## Requirements
Requires `numpy >= 1.16.1`, `scikit-learn >= 0.20.0`, `simplegp >= 0.9.9`.

## Reproducing the paper
Here's how to reproduce one run for a dataset "mydataset.csv"
```
cd Reproduce
python run.py mydataset
```
Please find all datasets used in the paper in `Reproduce/Datasets`.
Of course, perform multiple runs to obtain the same statistics of the paper.