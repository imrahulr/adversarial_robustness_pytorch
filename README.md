# Adversarial Robustness

This repository contains the unofficial implementation of the papers "[Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples](https://arxiv.org/abs/2010.03593)" (Gowal et al., 2020) and "[Fixing Data Augmentation to Improve Adversarial Robustness](https://arxiv.org/abs/2103.01946)" (Rebuffi et al., 2021) in [PyTorch](https://pytorch.org/). 

## Requirements

The code has been implemented and tested with `Python 3.8.5` and `PyTorch 1.8.0`.  To install the required packages:
```bash
$ pip install -r requirements.txt
```

## Usage

### Training 

Run [`train-wa.py`](./train-wa.py) for reproducing the results reported in the papers. For example, train a WideResNet-28-10 model via [TRADES](https://github.com/yaodongyu/TRADES) on CIFAR-10 with the additional pseudolabeled data provided by [Carmon et al., 2019](https://github.com/yaircarmon/semisup-adv) or the synthetic data from [Rebuffi et al., 2021](https://arxiv.org/abs/2103.01946) (without CutMix):

```
$ python train-wa.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc <name_of_the_experiment> \
    --data cifar10s \
    --batch-size 1024 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.4 \
    --beta 6.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename <path_to_additional_data>
```

**Note**: Note that with [Gowal et al., 2020](https://arxiv.org/abs/2010.03593), expect about 0.5% lower robust accuracy than that reported in the paper since the original implementation uses a custom regenerated pseudolabeled dataset which is not publicly available (See Section 4.3.1 [here](https://arxiv.org/abs/2010.03593)).

### Robustness Evaluation

The trained models can be evaluated by running [`eval-aa.py`](./eval-aa.py) which uses [AutoAttack](https://github.com/fra31/auto-attack) for evaluating the robust accuracy. For example:

```
$ python eval-aa.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc <name_of_the_experiment>
```

For PGD evaluation:
```
$ python eval-adv.py --wb --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc <name_of_the_experiment>
```

## Reference & Citing this work

If you use this code in your research, please cite the original works [[Paper](https://arxiv.org/abs/2010.03593)] [[Code in JAX+Haiku](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness)] [[Pretrained models](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness)]:

```
@article{gowal2020uncovering,
    title={Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples},
    author={Gowal, Sven and Qin, Chongli and Uesato, Jonathan and Mann, Timothy and Kohli, Pushmeet},
    journal={arXiv preprint arXiv:2010.03593},
    year={2020},
    url={https://arxiv.org/pdf/2010.03593}
}
```

*and/or*

```
@article{rebuffi2021fixing,
  title={Fixing Data Augmentation to Improve Adversarial Robustness},
  author={Rebuffi, Sylvestre-Alvise and Gowal, Sven and Calian, Dan A. and Stimberg, Florian and Wiles, Olivia and Mann, Timothy},
  journal={arXiv preprint arXiv:2103.01946},
  year={2021},
  url={https://arxiv.org/pdf/2103.01946}
}
```

*and* this repository:

```
@misc{rade2021pytorch,
    title = {{PyTorch} Implementation of Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples},
    author = {Rade, Rahul},
    year = {2021},
    url = {https://github.com/imrahulr/adversarial_robustness_pytorch}
}
```
