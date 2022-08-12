## FedPU &mdash; Official PyTorch Implementation

<!-- Official pytorch implementation of the paper "[APB2FACE: AUDIO-GUIDED FACE REENACTMENT WITH AUXILIARY POSE AND BLINK SIGNALS, ICASSP'20](https://arxiv.org/pdf/2004.14569.pdf)". -->

For any inquiries, please contact Xinyang Lin at [810427220@qq.com](mailto:810427220@qq.com)

ICML2022 - [Federated Learning with Positive and Unlabeled Data](https://arxiv.org/pdf/2106.10904.pdf)

## Using the Code

### Requirements

This code has been developed under `Python3.7`, `PyTorch 1.7.0` and `CUDA 11.0` on `Red Hat 8.3`. 


### Training

Train **FedPU** model. We implemented the dataloader of [FedMatch](https://arxiv.org/pdf/2108.05069.pdf)(ICLR 2021) on cifar10, for easier comparison.
 
FedPU works with FedAvg on non-iid data:
```shell
sh train_c10_FedAvg_FedPU_fmloader_noniid.sh
```

FedPU works with FedProx on non-iid data:
```shell
sh train_c10_FedProx_FedPU_fmloader_noniid.sh
```

Supervised learning experiment can be performed:
```shell
sh train_c10_FedAvg_SL_fmloader_noniid.sh
```


### Citation
If our work is helpful for your research, please consider citing:
```
@inproceedings{lin2022federated,
  title={Federated Learning with Positive and Unlabeled Data},
  author={Lin, Xinyang and Chen, Hanting and Xu, Yixing and Xu, Chao and Gui, Xiaolin and Deng, Yiping and Wang, Yunhe},
  booktitle={International Conference on Machine Learning},
  pages={13344--13355},
  year={2022},
  organization={PMLR}
}
```
