## 目前进展
- 由于原repo使用的版本比较久远，所以没有把原repo中的pytorch代码跑通，无法进行对比写log。
- 转换为paddle实现后，在Cifar10数据集上直接进行了训练。目前的训练结果，使用KNN classifiers, NCE m=4096, Top-1的准确率为79.90，与验收标准的80.40有小小的差别。


## Unsupervised Feature Learning via Non-parameteric Instance Discrimination

This repo constains the pytorch implementation for the CVPR2018 unsupervised learning paper [(arxiv)](https://arxiv.org/pdf/1805.01978.pdf).

## Updated Pretrained Model

An updated instance discrimination model with memory bank implementation and with nce-k=65536 negatives is provided.
The updated model is trained with Softmax-CE loss as in CPC/MoCo instead of the original NCE loss.

- [ResNet 50](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/lemniscate/lemniscate_resnet50_update.pth) (Linear ImageNet Acc 58.5%)


**Oldies**: original releases of ResNet18 and ResNet50 trained with 4096 negatives and the NCE loss.
Each tar ball contains the feature representation of all ImageNet training images (600 mb) and model weights (100-200mb).
You can also get these representations by forwarding the network for the entire ImageNet images.

- [ResNet 18](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/lemniscate/lemniscate_resnet18.pth) (top 1 nearest neighbor accuracy 41.0%)
- [ResNet 50](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/lemniscate/lemniscate_resnet50.pth) (top 1 nearest neighbor accuracy 46.8%)


## Highlight

- We formulate unsupervised learning from a completely different non-parametric perspective.
- Feature encodings can be as compact as 128 dimension for each image.
- Enjoys the benefit of advanced architectures and techniques from supervised learning.
- Runs seamlessly with nearest neighbor classifiers.

## Nearest Neighbor

Please follow [this link](http://zhirongw.westus2.cloudapp.azure.com/nn.html) for a list of nearest neighbors on ImageNet.
Results are visualized from our ResNet50 model, compared with raw image features and supervised features.
First column is the query image, followed by 20 retrievals ranked by the similarity.

## Usage

Our code extends the pytorch implementation of imagenet classification in [official pytorch release](https://github.com/pytorch/examples/tree/master/imagenet). 
Please refer to the official repo for details of data preparation and hardware configurations.

- supports python38 and [paddle=2.2.0rc]

- if you are looking for pytorch 0.3, please switch to tag v0.3

- clone this repo: `git clone https://github.com/vcowwy/lemniscate_paddle`

- During training, we monitor the supervised validation accuracy by K nearest neighbor with K=1, as it's faster, and gives a good estimation of the feature quality.

- Training on CIFAR10:

  `python cifar.py --nce-k 0 --nce-t 0.1 --lr 0.03`

