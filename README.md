# recoXplainer

As recommender systems today play an important role in our online experience and are involved in a wide range of 
decisions, multiple stakeholders demand explanations for the corresponding algorithmic predictions. 
These demands---together with the benefits of explanations (e.g., trust, efficiency, and sometimes even persuasion)--- 
have triggered significant interest from researchers in academia and industry. 

Nonetheless, to the best of our knowledge, no comprehensive toolkit for explainable recommender systems is available 
to the community yet. 
Instead, researchers are frequently faced with the challenge of re-implementing prior algorithms when creating and 
evaluating new approaches.
Aiming to address the resulting need, we introduce __RecoXplainer__, a software toolkit 
that includes several state-of-the-art explainability methods, and two evaluation metrics. 


## Install

### Pre-requirements
The following toolkits are necessary: 
- conda
- git

### Clone and environment set-up
Clone the repo:

```buildoutcfg
git clone https://github.com/ludovikcoba/recoxplainer.git
```

Create environment on conda:

```buildoutcfg
conda create -n recoxplainer python=3.6 
```

RecoXplainer was developed with python 3.6. 
Activate the new environment:

```buildoutcfg
conda activate recoxplainer
```

### Dependencies

Install torch as explained in https://pytorch.org/, we are using the version without CUDA.

When torch is installed navigate to the folder where you cloned the library and run:

```buildoutcfg
pip install -r requirements.txt
```
This command will install all the dependencies.
Next, install the _recoxplainer_:
```buildoutcfg
pip install -e .
```
And finally run the notebooks:

```buildoutcfg
jupyter notebook
```

### Running times

Running times for pre-processing, training, recommendation, explanation, and evaluation can be found in the `running_times.cvs' file. The reported time was calculated using a MacBook Pro 2,3 GHz Dual-Core Intel Core i5, 8 GB 2133 MHz LPDDR3.

