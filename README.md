# SparseGPT

This repository contains code to reproduce the key results of the paper [SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot](https://arxiv.org/abs/2301.00774).

Specifically, it provides scripts and implementations to:

* Evaluate baseline and pruned models on raw-WikiText2, PTB and C4-subset. (`datautils.py`, `opt.py`, `bloom.py`) 
* Perform unstructured, n:m and sparse + quantized SparseGPT compression on OPT and BLOOM models. (`sparsegpt.py`, `opt.py`, `bloom.py`)

We note that this SparseGPT implementation is based on our open-source [GPTQ code](https://github.com/IST-DASLab/gptq). 

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2
* `datasets`: tested on v1.17.0

## Usage

Here are some sample commands to run baselines and sparsification on OPT models, followed by perplexity evaluations on raw-WikiText2, PTB and C4.
See also the CMD-argument documentation.

```
# Run dense baseline
python opt.py facebook/opt-125m c4

# Run magnitude baseline
python opt.py facebook/opt-125m c4 --sparsity .5 --gmp

# Prune to 50\% uniform sparsity with SparseGPT
python opt.py facebook/opt-125m c4 --sparsity .5

# Prune to full 2:4 sparsity with SparseGPT
python opt.py facebook/opt-125m c4 --prunen 2 --prunem 4

# Prune to 50\% + 4-bit with SparseGPT
python opt.py facebook/opt-125m c4 --sparsity .5 --wbits 4
```

To run on other OPT models, replace "facebook/opt-125m" by the HuggingFace name of the corresponding model.
For the 175B model, access must first be requested from Meta and the checkpoint converted to HuggingFace format, then its location can simply be passed as a name to this script.

The BLOOM script `bloom.py` has a very similar interface, however some features are currently only available for OPT, e.g.:

```
# Sparsify BLOOM-176B with SparseGPT
python bloom.py bigscience/bloom c4 --sparsity .5
```

We also provide LLaMA pruning script with the very same interface:

```
# Sparsify LLaMa with SparseGPT
python llama.py LLAMA_HF_WEIGHTS_LOCATION c4 --sparsity 0.5
```

In case one would like to save the sparsified model specify path to saved checkpoint via  `--save` flag.

One can optionally log evalution results to W&B with `--log_wandb`. 

## Demo

One can try SparseGPT via the colab demo - `demo.ipynb`. 

## Cite

If you found this work useful, please consider citing:

```
@article{frantar-sparsegpt,
  title={{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot}, 
  author={Elias Frantar and Dan Alistarh},
  year={2023},
  journal={arXiv preprint arXiv:2301.00774}
}
```
