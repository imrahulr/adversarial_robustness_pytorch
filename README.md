# Adversarial Robustness

This repository contains the unofficial implementation of the paper "[Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples](https://arxiv.org/abs/2010.03593)" (Gowal et al., 2020) in [PyTorch](https://pytorch.org/). 

## Requirements

The code has been implemented and tested with `Python 3.8.5` and `PyTorch 1.8.0`.  To install the required packages:
```bash
$ pip install -r requirements.txt
```

## Usage

### Training 

Run [`train-wa.py`](./train-wa.py) for reproducing the results reported in the paper. For example, train a WideResNet-28-10 model via [TRADES](https://github.com/yaodongyu/TRADES) on CIFAR-10 with the additional pseudolabeled data provided by [Carmon et al., 2019](https://github.com/yaircarmon/semisup-adv):

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

**Note**: Expect a 0.5-0.9% lower robust accuracy than that reported in the paper since the original implementation uses a custom regenerated pseudolabeled data which is not publicly available (See Section 4.3.1 [here](https://arxiv.org/abs/2010.03593)).

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

## Reference

[1]  Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples, Gowal et al., 2020. [[Paper](https://arxiv.org/abs/2010.03593)] [[Code in JAX+Haiku](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness)] [[Pretrained models](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness)]