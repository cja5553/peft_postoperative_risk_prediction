
# tipeft

**T**abular-**i**nfused **P**arameter **E**fficient **F**ine**t**uning (tipeft) is a novel PEFT method designed to infuse tabular features into the initialization process of re-parameterization parameter efficient finetuning (PEFT) methods. This provides an element of well-informed and representational capacity towards the newly introduced PEFT parameters, which are usually introduced and initialized independently

![Overview of tipeft framework](Figure_1.jpg)

It is specifically designed for postoperative predictions in clinical care, where predictive and valuable pre-operative tabular features are often under-utilized in language model finetuning. For now, it supports both `LoRA` and `IA3`


## Requirements  
### Dependencies


The following Python packages are required for `tipeft`:

- `torch`
- `transformers`
- `peft`
- `accelerate`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`

Install dependencies with:

```bash
pip install torch transformers peft accelerate numpy pandas scikit-learn tqdm
```

#### Note on Pytorch installation
Because PyTorch wheels vary by CUDA version and hardware, it is recommended to install PyTorch manually following the instructions at:
https://pytorch.org/ 

### System Requirements

`tipeft` has been tested and verified on the following configuration:

| Component | Tested Version |
|-----------|----------------|
| OS | Windows 10 |
| Python | 3.9.19 |
| CUDA | 12.6 |

#### Important Notes

- **Environment**: Must be run in a Jupyter notebook. Running as a standalone Python script may cause multiprocessing issues.
- **CPU cores**: At least 10 CPU cores recommended (uses `Pool(processes=10)` internally).
- **GPU**: CUDA-compatible GPU required.
- **OS**: Tested on Windows. Linux/Mac compatibility not yet verified.

#### Known Compatibility Limitations

1. **Jupyter only** - Uses `tqdm.notebook` which may not display correctly outside Jupyter.
2. **Multiprocessing** - May behave differently on Linux/Mac due to different multiprocessing backends.

If you encounter issues on a different setup, please open an issue with your system info.

#### GPU requirements

`tipeft` is designed for GPU acceleration.
- At least 1 GPU is recommended
- Suggested minimum: 16GB VRAM 
- Memory usage depends on:
    - sequence length
    - model size
    - batch size
    - peft configuration



## Installation
To install in python, simply do the following: 
```bash
pip install tipeft
```


## Quick Guide


