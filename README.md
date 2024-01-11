# Model Agnostic Explainable Selective Regression via Uncertainty Estimation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper titled "Model Agnostic Explainable Selective Regression via Uncertainty Estimation". In this work, we propose a novel approach to selective regression that leverages model-agnostic, non-parametric uncertainty estimation. Our framework demonstrates superior performance when compared to state-of-the-art selective regressors. We conduct extensive benchmarking on 69 datasets to showcase the effectiveness of our approach. Additionally, we apply explainable AI techniques to gain insights into the factors driving selective regression.

## Abstract

With the widespread adoption of machine learning techniques, the requirements for predictive models have expanded beyond mere high performance. Often, it is necessary for an estimator not only to provide predictions but also to abstain from making predictions in uncertain or ambiguous situations. This concept is referred to as selective prediction. While selective prediction for classification tasks has received substantial attention, the challenge of selective regression remains relatively unexplored.

In this paper, we introduce an innovative approach to selective regression, which employs a model-agnostic and non-parametric uncertainty estimation strategy. Our proposed framework showcases significant improvements in performance when compared to existing state-of-the-art selective regression methods. We substantiate the efficacy of our approach through an extensive benchmarking process involving 69 diverse datasets.

To enhance the interpretability of our selective regression approach, we incorporate explainable AI techniques. These techniques enable us to gain a deeper understanding of the underlying drivers behind the selective regression process.

## Repository Structure

- `pmlbBenchmark.py`: code to reproduce the experiments of Q1-Q2 using pmlbBenchmark data.
- `TabGrinBenchmark.py`: code to reproduce the experiments of Q1-Q2 using data from (Grinsztajn et al., 2023).
- `xAIExp.py`: code to reproduce the experiments of Q3.
- `tools/`: Additional files to run experiments, if any, related to the package and experiments.
- `LICENSE`: The license information for the contents of this repository.
- `README.md`: You are here!

## Installation

Install the required package using:


```
$ pip install -r requirements.txt
```



## Usage

The experiments for Q1-Q2 can be reproduced by running the following commands, using as $SEED the values 42, 999, 12345, 291091, 123456789:

```
$ python pmlbBenchmark.py  --meta doubt doubtNew mapieBase plugin gold scross --reg xgb lgbm rf dt lr --seed $SEED
$ python TabGRinBenchmark.py  --meta doubt doubtNew mapieBase plugin gold scross --reg xgb lgbm rf dt lr --seed $SEED
```

> For Q3, run the following:

```
$ python xAIExp.py
```


## Citation

If you find our work useful in your research, please consider citing our paper:


## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

For any questions, issues, or collaborations, please feel free to contact us.


