## Dependencies

- Python >= 3.7
- [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [RNA](https://github.com/ViennaRNA/ViennaRNA)

## Installing

1) (optional) Create a virtual environment to install the tool

2) Install CUSTOM and its requirements using pip:
```bash
pip install git+https://github.com/suresh-pokharel/CUSTOM.git
```

## Basic usage

As a basic example, here is the code to optimize an eGFP protein to kidney:
```python
# Import package
import custom
# Start the optimizer
opt = TissueOptimizer("Kidney", n_pool=1000)
# Optimize the eGFP sequence
egfp = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
opt.optimize(egfp)
# Select the top 10 sequences
best_egfp_lung = opt.select_best(by={"MFE":"min","MFEini":"max","CAI":"max","CPB":"max","UD":0.15, "GC":0.5},homopolymers=7, top=10)
```
