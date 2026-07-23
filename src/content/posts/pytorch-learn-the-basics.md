---
title: "从 Tensor 到 Checkpoint：补上 PyTorch 的完整工作流"
description: 一遍 PyTorch 官方 Learn the Basics 的学习记录，串起 Tensor、数据加载、Autograd、优化循环与 checkpoint。
date: 2026-07-23
category: 学习笔记
categoryCode: STUDY
series: PyTorch 学习
tags: [PyTorch, 深度学习, 学习笔记]
kind: article
draft: false
---

我学习该教程作为对 PyTorch 学习的补充。之前学过 AI 引论， CS 231 n，CV 导论都使用过 PyTorch，但是作业都是下发有一个完整的框架，只需要自己完成其中部分函数，所以对整个 PyTorch 的工作流并不熟悉。

在正式开始暑研前，我花时间过了一遍官方的 [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) 教程。借此机会了解 PyTorch 代码并且顺便复习基础深度学习知识。

本文使用 codex 进行整理润色。

```text
原始数据
  -> Dataset 定义一个样本如何取得
  -> Transform 把样本变成模型需要的形式
  -> DataLoader 组织 batch
  -> nn.Module 完成前向计算
  -> loss 衡量预测误差
  -> autograd 计算梯度
  -> optimizer 更新参数
  -> state_dict / checkpoint 保存结果或训练状态
```

## 1. Tensor：PyTorch 中的数据载体

Tensor 和 NumPy 的 `ndarray` 很像。它们都表示多维数组，也都支持索引、切片、矩阵运算和广播。Tensor 多出的两项能力是：

- 可以放在 CPU、GPU 或其他加速器上计算。
- 可以被 autograd 记录，用于自动求导。

模型的输入、输出、参数和梯度最终都以 Tensor 表示。

### 1.1 创建 Tensor

```python
import numpy as np
import torch

data = [[1, 2], [3, 4]]

# 从 Python 数据创建，dtype 会根据数据推断
x_data = torch.tensor(data)

# 从 NumPy 数组创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 继承另一个 Tensor 的 shape 和 dtype
x_ones = torch.ones_like(x_data)

# 显式覆盖 dtype
x_rand = torch.rand_like(x_data, dtype=torch.float32)

# 按 shape 创建
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```

`torch.tensor(np_array)` 通常会复制数据，而 `torch.from_numpy(np_array)` 会与 NumPy 数组共享底层内存。共享内存时，修改一方可能影响另一方。如果需要互不影响的副本，可以再调用 `.clone()`。

### 1.2 三个最常检查的属性

```python
tensor.shape
tensor.dtype
tensor.device
```

- `shape` 描述每个维度的长度。
- `dtype` 描述每个元素的存储和计算类型。
- `device` 描述数据存放和计算的位置。

深度学习代码里常见的 dtype 有：

| 类别 | 常见 dtype | 典型用途 |
|---|---|---|
| 浮点数 | `float32` | 默认训练和通用计算 |
| 低精度浮点数 | `float16`、`bfloat16` | 混合精度训练和推理 |
| 双精度浮点数 | `float64` | 对精度要求较高的数值计算 |
| 整数 | `int64`、`int32` | 分类标签、索引和计数 |
| 无符号整数 | `uint8` | 原始图像像素和字节数据 |
| 布尔值 | `bool` | mask 和条件索引 |
| 复数 | `complex64`、`complex128` | 复数计算 |

常见 device 有 `cpu`、`cuda` 和 `mps`。`cuda:0` 表示第 0 张 NVIDIA GPU，`mps` 表示 Apple Silicon 上的 Metal 后端。`meta` 是一种特殊设备，只记录 Tensor 的元信息而不分配真实数据，常用于大模型初始化和形状推导。

`.to(...)` 可以转换 dtype 或迁移 device：

```python
x = x.to(torch.bfloat16)
x = x.to("cuda:0")
```

设备迁移通常会复制数据，因此不要在训练循环里无意义地反复搬运大 Tensor。

新版 PyTorch 还提供了 `torch.accelerator`，用统一接口检查当前可用的加速器：

```python
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
```

`torch.accelerator` 负责选择和管理加速器，Tensor 最终仍然通过自己的 `device` 属性表示所处位置。

### 1.3 常用运算

Tensor 的索引和切片与 NumPy 基本一致：

```python
tensor = torch.ones(4, 4)

first_row = tensor[0]
first_column = tensor[:, 0]
last_column = tensor[..., -1]
tensor[:, 1] = 0
```

拼接时需要区分 `cat` 和 `stack`：

```python
x = torch.ones(2, 3)

torch.cat([x, x], dim=0).shape    # [4, 3]，沿已有维度拼接
torch.stack([x, x], dim=0).shape  # [2, 2, 3]，创建一个新维度
```

矩阵乘法和逐元素乘法不是一回事：

```python
y = x @ x.T          # 矩阵乘法
y = torch.matmul(x, x.T)

z = x * x            # 对应位置相乘
z = torch.mul(x, x)
```

只有一个元素的 Tensor 可以用 `.item()` 转成 Python 数值：

```python
total = tensor.sum()
total_value = total.item()
```

`.item()` 会脱离计算图，而且如果 Tensor 在 GPU 上，还会触发一次设备同步，因此不适合在高频计算路径里滥用。

### 1.4 原地操作

方法名末尾带 `_` 的操作通常是原地操作，例如：

```python
x.add_(1)
x.mul_(2)
x.zero_()
```

`x += 1` 通常也是原地修改，而 `x = x + 1` 会计算一个新 Tensor，再让变量 `x` 指向它。

原地操作有时能少分配一块内存，但不保证一定更快。更麻烦的是，autograd 可能需要旧值来计算反向传播。如果旧值被覆盖，PyTorch 会报错，或者阻止对需要梯度的 leaf Tensor 进行原地修改。因此，在没有明确内存收益和正确性证明时，我倾向于先写非原地版本。

## 2. Dataset、Transform 和 DataLoader

这三个组件把数据处理与模型训练分开：

```text
index
  -> Dataset.__getitem__(index)
  -> (feature, label)
  -> transform / target_transform
  -> DataLoader 组合多个样本
  -> (batch_features, batch_labels)
```

### 2.1 Dataset：定义一个样本

`Dataset` 负责回答两个问题：

- 数据集有多少个样本？
- 给定索引 `idx`，如何得到第 `idx` 个样本？

自定义 Dataset 通常实现三个方法：

```python
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=None,
        target_transform=None,
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir,
            self.img_labels.iloc[idx, 0],
        )
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
```

`__init__` 在创建 Dataset 时运行一次，适合保存文件路径和标注表。`__getitem__` 在取样本时运行，因此大型图像数据集不必一次性全部读入内存。

### 2.2 Transform：把样本转换成模型需要的形式

`transform` 处理 feature，`target_transform` 处理 label。例如，FashionMNIST 的原始图像需要转换为浮点 Tensor：

```python
from torchvision import datasets
from torchvision.transforms import v2

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]),
)
```

`scale=True` 不只是修改 dtype，还会把整数图像值缩放到适合浮点表示的范围。

官方 Transform 教程还演示了把类别编号转换成 one-hot：

```python
import torch.nn.functional as F

target_transform = v2.Lambda(
    lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
)
```

这段代码用于展示 `target_transform`。后面的 FashionMNIST 训练使用 `nn.CrossEntropyLoss` 时，可以直接保留 `int64` 类别编号，不需要先转 one-hot。loss 所要求的 target 形状和 dtype 必须单独确认。

### 2.3 DataLoader：组织 batch

`DataLoader` 接收 Dataset，然后负责批处理、打乱和并行加载：

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    training_data,
    batch_size=64,
    shuffle=True,
)
```

如果单个图像的 shape 是 `[1, 28, 28]`，batch size 为 64，那么一批图像通常是 `[64, 1, 28, 28]`，标签是 `[64]`。

`shuffle=True` 一般用于训练集。验证集和测试集通常不需要打乱。`num_workers` 可以并行准备样本，但最佳值取决于运行环境。在 Colab 或不同操作系统上，盲目增大 worker 数量不一定更快。

我目前对二者最简洁的理解是：

> Dataset 决定如何取一个样本，DataLoader 决定如何迭代并组合这些样本。

## 3. 用 `nn.Module` 搭建模型

PyTorch 中的神经网络继承 `nn.Module`。一个 Module 可以包含层，也可以包含其他 Module，因此模型天然具有嵌套结构。

```python
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

`__init__` 注册模型包含的层，`forward` 描述输入如何流过这些层。使用模型时应该写 `model(x)`，不要直接调用 `model.forward(x)`，因为 `nn.Module.__call__` 还负责 hooks 等额外机制。

这段网络中的 shape 变化是：

```text
[B, 1, 28, 28]
  -> Flatten
[B, 784]
  -> Linear(784, 512)
[B, 512]
  -> Linear(512, 512)
[B, 512]
  -> Linear(512, 10)
[B, 10] logits
```

最后的 10 个值是 logits，不是概率。训练时可以直接把 logits 传给 `nn.CrossEntropyLoss`，因为它内部已经组合了 `LogSoftmax` 和 `NLLLoss`。

### 3.1 Parameter 及其遍历

`nn.Linear` 的 weight 和 bias 是 `nn.Parameter`。Parameter 是一种默认需要梯度、并能被 Module 自动注册的 Tensor。

```python
for parameter in model.parameters():
    print(parameter.shape)

for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.requires_grad)
```

- `.parameters()` 返回 Parameter。
- `.named_parameters()` 返回 `(name, Parameter)`。

优化器通常接收 `model.parameters()`：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3,
)
```

如果要按名称冻结 backbone、只训练 head，`.named_parameters()` 更方便：

```python
for name, parameter in model.named_parameters():
    parameter.requires_grad = name.startswith("classifier")
```

需要注意，`state_dict()` 不只包含 Parameter，还包含 BatchNorm 的 `running_mean` 一类需要持久化的 buffer。

## 4. Autograd：梯度如何流过计算图

训练的目标是计算 loss 对模型参数的导数。PyTorch 在前向计算时动态记录运算关系，`loss.backward()` 再沿计算图反向应用链式法则。

```python
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

logits = x @ w + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(
    logits,
    y,
)

loss.backward()

print(w.grad)
print(b.grad)
```

### 4.1 `requires_grad`、leaf 和 `.grad`

下面四件事需要分开：

- 是否追踪梯度，由 `requires_grad` 决定。
- 是否由用户直接创建，而不是由某次运算产生，由 `is_leaf` 描述。
- 是否参与反向传播，取决于它是否位于需要梯度的计算路径上。
- 是否把结果保存在 `.grad`，与是否算过这个梯度不是一回事。

| Tensor | `requires_grad` | 参与反向传播 | 默认保存 `.grad` |
|---|---:|---:|---:|
| leaf | `False` | 否 | 否 |
| leaf | `True` | 是 | 是 |
| non-leaf | `True` | 是 | 否 |

non-leaf Tensor 的梯度会在链式法则中经过，但默认不保存在 `.grad`。调试时可以显式保留：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
y.retain_grad()
z = y**2

z.backward()

print(x.grad)  # 36
print(y.grad)  # 12
```

`retain_grad()` 不是让 Tensor 开始求导，而是让 autograd 不丢弃这个中间梯度。

由运算产生的 non-leaf Tensor 通常有 `grad_fn`，它指向反向传播时应调用的函数。用户创建的 leaf Tensor 没有生成它的运算，所以 `grad_fn` 通常是 `None`。

### 4.2 标量 backward 与向量积

如果 `loss` 是标量，可以直接调用：

```python
loss.backward()
```

如果输出 `y` 不是标量，需要提供一个与 `y` 同 shape 的向量 `v`：

```python
y.backward(v)
```

这不是因为 PyTorch 无法计算 Jacobian，而是反向模式自动微分通常不显式构造完整 Jacobian。`backward(v)` 直接计算 vector-Jacobian product，也就是把上游梯度 `v` 沿计算图传回来。对深度学习中的标量 loss 来说，上游梯度默认就是 1。

### 4.3 停止记录或切断计算图

```python
detached = tensor.detach()
```

`detach()` 返回一个不再连接原计算图的 Tensor。它通常仍与原 Tensor 共享存储，因此原地修改时仍要小心。

不需要梯度的代码块可以写成：

```python
with torch.no_grad():
    predictions = model(X)
```

纯推理还可以使用：

```python
with torch.inference_mode():
    predictions = model(X)
```

`inference_mode` 关闭的 autograd bookkeeping 更多，限制也更强。训练中的验证循环通常使用 `no_grad()` 就已经足够清楚。

一次 `backward()` 后，PyTorch 通常会释放本次反向传播不再需要的图。如果确实需要在同一张图上多次 backward，要理解原因后再使用 `retain_graph=True`，而不是把它当作常规写法。

## 5. 优化循环：从梯度到参数更新

模型训练是重复执行以下过程：

```text
取一个 batch
  -> 前向计算 logits
  -> 计算 loss
  -> backward 得到 parameter.grad
  -> optimizer.step 更新参数
  -> 清理梯度
```

### 5.1 loss 和 optimizer

分类任务常用 `nn.CrossEntropyLoss`，回归任务常用 `nn.MSELoss`。对于 FashionMNIST：

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3,
)
```

学习率、batch size 和 epoch 数量都是超参数：

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

### 5.2 一套可运行的训练和测试循环

```python
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * len(X)
            print(
                f"loss: {loss.item():>7f} "
                f"[{current:>5d}/{size:>5d}]"
            )


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (
                (pred.argmax(dim=1) == y)
                .type(torch.float32)
                .sum()
                .item()
            )

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error:\n"
        f"Accuracy: {100 * correct:>0.1f}%, "
        f"Avg loss: {test_loss:>8f}"
    )
```

`model.train()` 和 `model.eval()` 不负责打开或关闭梯度。它们切换 Dropout、BatchNorm 等模块的行为。`torch.no_grad()` 才负责在测试循环中关闭梯度记录。

### 5.3 梯度为什么要清理

`backward()` 会把新梯度累加到 Parameter 的 `.grad`，而不是自动覆盖旧值：

```python
parameter.grad += new_gradient
```

因此每次独立参数更新前都要清理上一轮梯度。可以把 `optimizer.zero_grad()` 放在 backward 前，也可以像部分官方代码那样放在 `optimizer.step()` 后，只要每个更新周期的边界一致。

当前 PyTorch 中，`optimizer.zero_grad()` 默认通常把 `.grad` 设为 `None`，而不是给现有梯度 Tensor 填 0。下一次 backward 会重新创建梯度。这与手动调用 `grad.zero_()` 在内存和部分边界行为上有所区别。

梯度属于 Parameter，不属于 optimizer。假设 SGD 和 Adam 管理同一个 Parameter：

```python
sgd.zero_grad()
```

会清理这些 Parameter 的 `.grad`，所以 Adam 随后看到的也是空梯度。但 Adam 的一阶、二阶动量等 optimizer state 不会被清理。让两个 optimizer 对同一参数都调用 `step()` 会连续更新参数两次，除非算法明确需要，否则应避免这种设计。

### 5.4 梯度累计

显存只能容纳较小 micro-batch 时，可以先做多次 backward，再做一次参数更新：

```python
accumulation_steps = 4
optimizer.zero_grad()

for step, (X, y) in enumerate(train_dataloader):
    X = X.to(device)
    y = y.to(device)

    pred = model(X)
    loss = loss_fn(pred, y) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

在 mean loss、micro-batch 大小相同且累计期间参数固定等条件下，除以 `accumulation_steps` 后得到的是这些 micro-batch 梯度的平均值。若最后剩余的 micro-batch 数量不足、样本数不相等，或者模型含有依赖 batch 统计量的层，还要单独处理，不能只照抄这个简化版本。

## 6. 保存权重、加载模型与 checkpoint

### 6.1 `state_dict`

模型把 Parameter 和持久化 buffer 放在 `state_dict` 中。最常见的保存方式是：

```python
torch.save(
    model.state_dict(),
    "model_weights.pth",
)
```

加载时要先创建结构相同的模型，再把 Tensor 复制到对应参数：

```python
model = NeuralNetwork()

state_dict = torch.load(
    "model_weights.pth",
    weights_only=True,
    map_location="cpu",
)
model.load_state_dict(state_dict)
model.eval()
```

`model.eval()` 很容易遗漏。即使权重已经正确加载，Dropout 和 BatchNorm 在训练模式与评估模式下仍有不同表现。

### 6.2 `.pt` 和 `.pth` 是什么

`.pt` 和 `.pth` 是社区约定的文件后缀，后缀本身不决定内容。`torch.save(obj, path)` 保存的是传入对象的序列化结果，既可以是 Tensor，也可以是字典、列表或其他可 pickle 的 Python 对象。

更准确地说，现代 `torch.save` 默认写出一个未压缩的 ZIP64 archive。对象元信息通过 pickle 保存，Tensor storage 会单独存放。因此，把 `torch.save` 理解成"序列化到二进制文件"是合适的，但称为"压缩"并不准确。

`torch.load` 底层涉及 unpickling，不应加载来源不可信的文件。加载普通权重字典时使用 `weights_only=True`，可以限制反序列化过程中允许构造的对象类型，但它也不是对恶意文件的完整沙箱。

### 6.3 结构对不上会怎样

`load_state_dict(..., strict=True)` 默认要求 key 完整匹配。如果 checkpoint 多了一些 key，或者当前模型缺了一些 key，会报出 `missing_keys` 和 `unexpected_keys`。

迁移学习时可以使用：

```python
result = model.load_state_dict(
    state_dict,
    strict=False,
)
print(result.missing_keys)
print(result.unexpected_keys)
```

`strict=False` 可以忽略缺失或多余的 key，但不会自动解决同名参数 shape 不一致。例如旧分类头是 `[1000, 768]`，新分类头是 `[10, 768]`，仍然需要删除这些不兼容权重，或者在加载前按名称和 shape 过滤：

```python
current = model.state_dict()

compatible = {
    key: value
    for key, value in state_dict.items()
    if key in current and value.shape == current[key].shape
}

result = model.load_state_dict(
    compatible,
    strict=False,
)
```

这种情况下，新分类头保留自己的随机初始化。

### 6.4 checkpoint 里放什么

checkpoint 没有固定 schema。它通常就是一个由训练代码定义的字典，目标是保存足够的信息，使训练能够从中断处继续。

```python
checkpoint = {
    "epoch": epoch,
    "global_step": global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": config,
}

torch.save(checkpoint, "checkpoint.pth")
```

加载时先按相同配置创建对象：

```python
checkpoint = torch.load(
    "checkpoint.pth",
    weights_only=True,
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)
scheduler.load_state_dict(
    checkpoint["scheduler_state_dict"]
)

start_epoch = checkpoint["epoch"] + 1
global_step = checkpoint["global_step"]
```

不同目的需要保存的内容不同：

| 目的 | 通常需要的内容 |
|---|---|
| 推理或发布权重 | 模型 `state_dict` 和模型配置 |
| 微调初始化 | 模型 `state_dict`，有时只保存部分模块 |
| 断点续训 | 模型、optimizer、scheduler、epoch 和 global step |
| 混合精度续训 | 再加 GradScaler 状态 |
| 尽量严格复现 | 再加 Python、NumPy、CPU/GPU RNG 状态及数据迭代位置 |
| 多 optimizer 训练 | 分别保存每个 optimizer 的 `state_dict` |

loss 数值和 loss history 可以为了记录实验而保存，但恢复训练真正依赖的是参数、优化器状态、学习率进度和训练位置。只恢复模型权重而不恢复 Adam 的动量，虽然能继续运行，却不是原训练轨迹的无缝延续。

## 7. 我现在如何理解完整工作流

经过这次学习，我对 PyTorch 的理解不再是几个散落的函数。它更像一组分工明确、通过 Tensor 连接起来的对象：

```text
Dataset
  __getitem__ 返回一个样本

DataLoader
  把样本组成 batch

nn.Module
  用 Parameter 和层计算 logits

loss_fn
  把 logits 与 target 变成标量 loss

autograd
  把 loss 的梯度传回 Parameter.grad

optimizer
  读取 Parameter.grad 并更新 Parameter

state_dict / checkpoint
  保存推理或恢复训练所需的状态
```

这里最容易混淆的几组概念是：

- Tensor 的梯度是否被计算，与 `.grad` 是否被保存不是一回事。
- 梯度属于 Parameter，optimizer 只持有参数引用和自己的更新状态。
- `model.eval()` 改变模块行为，`torch.no_grad()` 关闭梯度记录，两者作用不同。
- `state_dict` 主要描述模型状态，checkpoint 描述一次训练如何继续。
- `.pt` 和 `.pth` 只是命名约定，真正的内容取决于传给 `torch.save` 的对象。

我接下来还想继续理解 PyTorch 更底层的部分，例如 Tensor 的 stride 和 view、DataLoader 的并行机制、autograd 如何实现反向图，以及 optimizer state 在混合精度和分布式训练中如何管理。这些问题留到之后的课程和实际代码中再逐步补上。

## 参考

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
