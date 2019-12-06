# The author's PyTorch implementation of Neural Painters

![banner](readme_img/banner.jpg)

Neural Painters: A learned differentiable constraint for generating brushstroke paintings

https://arxiv.org/abs/1904.08410

## Dependencies

Dependencies are listed in [`environment.yaml`](https://github.com/reiinakano/neural-painters-pytorch/tree/master/environment.yaml) but the notable ones are:

* PyTorch 1.3+
* kornia
* gdown
* (Optional) MyPaint - installing this is a bit of a pain but you can view examples of how to set up the exact versions in the notebooks [`train_vae_painter.ipynb`](https://colab.research.google.com/github/reiinakano/neural-painters-pytorch/blob/master/notebooks/train_vae_painter.ipynb) and [`train_gan_painter.ipynb`](https://colab.research.google.com/github/reiinakano/neural-painters-pytorch/blob/master/notebooks/train_gan_painter.ipynb).

## Notebooks

The best way to figure out how to use this code is to play around with the provided Colaboratory notebooks. We provide pre-trained neural painters.

There are runnable notebooks for the paper in the [`notebooks/`](https://github.com/reiinakano/neural-painters-pytorch/tree/master/notebooks) folder.

Since most people will probably only be interested in certain parts of the paper, we have designed them so you will be able to run each part as standalone notebooks. For example, we have provided pre-trained neural painters so you can run the style transfer notebook without having to train your own neural painter.

* [`train_vae_painter.ipynb`](https://colab.research.google.com/github/reiinakano/neural-painters-pytorch/blob/master/notebooks/train_vae_painter.ipynb) and [`train_gan_painter.ipynb`](https://colab.research.google.com/github/reiinakano/neural-painters-pytorch/blob/master/notebooks/train_gan_painter.ipynb) - These notebooks contain code to train VAE and GAN neural painters, respectively.

![vae](readme_img/vae_neural_painter_example.jpg)
![gan](readme_img/gan_neural_painter_example.jpg)

* [`visualizing_imagenet.ipynb`](https://colab.research.google.com/github/reiinakano/neural-painters-pytorch/blob/master/notebooks/visualizing_imagenet.ipynb) - Contains code for the "Visualizing ImageNet Classes" subsection of the paper. Requires a neural painter. We provide pre-trained neural painters if you don't want to train your own.

![visualize_imagenet](readme_img/mix.jpg)

* [`intrinsic_style_transfer.ipynb`](https://colab.research.google.com/github/reiinakano/neural-painters-pytorch/blob/master/notebooks/intrinsic_style_transfer.ipynb) - Contains code for the "Intrinsic Style Transfer" subsection of the paper. Requires a neural painter. We provide pre-trained neural painters if you don't want to train your own.

![intrinsic](readme_img/styletransferstaticdiagram.jpg)
