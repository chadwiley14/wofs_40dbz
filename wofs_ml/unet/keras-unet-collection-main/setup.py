import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "keras-unet-collection",
    version = "0.1.13",
    author = "Yingkai (Kyle) Sha",
    author_email = "yingkaisha@gmail.com",
    description = "The Tensorflow, Keras implementation of U-net, V-net, U-net++, UNET 3+, Attention U-net, R2U-net, ResUnet-a, U^2-Net, TransUNET, and Swin-UNET with optional ImageNet-trained backbones.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/yingkaisha/keras-unet-collection",
    packages = setuptools.find_packages(),
    classifiers=[ "Programming Language :: Python :: 3",
                  "License :: OSI Approved :: MIT License",
                  "Operating System :: OS Independent",],
    python_requires='>=3.6',)

