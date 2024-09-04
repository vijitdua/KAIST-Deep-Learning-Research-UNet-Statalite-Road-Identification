# Road Identification from Satellite Images using UNet & Attention UNet

---

(Disclaimer: README written by chat-gpt)

## What is this Project?

This project uses deep learning models, specifically UNet and Attention UNet, to identify road infrastructure from
satellite images. The goal is to automate the process of road mapping, minimizing the manual labor typically required.
The models were tested using satellite datasets from Massachusetts and SpaceNet (Vegas & Shanghai), allowing for
evaluation across diverse environments and conditions.

> This project was developed during my research at KAIST as part of the KISS 2024 Track 2 program. While the repository
> is being made public a few months after the initial development, it provides a clear overview of the models and methods
> used. Some of the code comes from forked repositories, which were adapted to meet the requirements of this research. 

> The code is unpolished due to the rapid pace of code development given prior in-experience (2-3 weeks) and no prior intentions of accessibility or open-sourcing.

---

## Research Paper

For detailed insights into the research, please check out the [Research Paper](./research-paper.pdf) included in this
repository.

---

## Tech Stack

### Models

- **UNet**: A convolutional neural network used for image segmentation.
- **Attention UNet**: A variation of the UNet model incorporating attention mechanisms for improved accuracy in
  segmentation tasks.

### Tools

- **Framework**: PyTorch
- **Datasets**: Massachusetts Road Dataset, SpaceNet Dataset (Vegas, Shanghai)
- **Environment**: Python 3.8

---


## Results & Other details

Check out the [Research Paper](./research-paper.pdf)

---