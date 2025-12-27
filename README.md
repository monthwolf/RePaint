# RePaint

> 以下是基于原英文文档的中文翻译版本 README，同时增加了 jupyter notebook 的使用说明。

**使用去噪扩散概率模型进行图像修复 (Inpainting)**

CVPR 2022 [[论文]](https://bit.ly/3b1ABEb)

[![Denoising_Diffusion_Inpainting_Animation](https://user-images.githubusercontent.com/11280511/150849757-5cd762cb-07a3-46aa-a906-0fe4606eba3b.gif)](#)

## 安装与设置

### 1. 获取代码

```bash
git clone https://github.com/andreas128/RePaint.git
```

### 2. 环境配置

```bash
pip install numpy torch blobfile tqdm pyYaml pillow    # 例如: torch 1.7.1+cu110.
```

### 3. 下载模型和数据

```bash
pip install --upgrade gdown && bash ./download.sh
```

该脚本会下载 ImageNet, CelebA-HQ, 和 Places2 的模型，以及人脸示例和示例遮罩 (masks)。

### 4. 运行示例 (命令行方式)

```bash
python test.py --conf_path confs/face_example.yml
```

输出结果可以在 `./log/face_example/inpainted` 中找到。

_注意: 代码重构后，我们没有重新评估所有实验。_

### 5. Jupyter Notebook 训练方式

建议直接运行提供的 `Repaint.ipynb` 文件。支持使用自定义数据集进行训练，同时训练会输出过程图在输出文件夹中，可以直观看到训练过程。

**使用步骤:**

1. 打开 `Repaint.ipynb`。
2. 按照 Notebook 中的单元格顺序执行：
   - 自动安装依赖库。
   - 下载必要的预训练模型。
   - 配置并运行推理测试。

<br>

# RePaint 利用扩散模型填充缺失的图像部分

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td><img alt="RePaint Inpainting using Denoising Diffusion Probabilistic Models Demo 1" src="https://user-images.githubusercontent.com/11280511/150766080-9f3d7bc9-99f2-472e-9e5d-b6ed456340d1.gif"></td>
        <td><img alt="RePaint Inpainting using Denoising Diffusion Probabilistic Models Demo 2" src="https://user-images.githubusercontent.com/11280511/150766125-adf5a3cb-17f2-432c-a8f6-ce0b97122819.gif"></td>
  </tr>
</table>

**蓝色的部分是什么?** <br>
那些是缺失的部分，需要由 RePaint 进行填充。<br> RePaint 根据已知部分生成缺失的内容。

**它是如何工作的?** <br>
RePaint 从纯噪声开始。然后逐步对图像进行去噪。<br> 在每一步中，它利用已知部分来填充未知部分。

**为什么生成过程中噪声水平会波动?** <br>
我们的噪声调度改进了生成部分与已知部分之间的协调性 [[4.2 重采样]](https://bit.ly/3b1ABEb)。

<br>

## 数据详情

**哪些数据集和遮罩有现成的配置文件?**

我们在 [`./confs`](https://github.com/andreas128/RePaint/tree/main/confs) 目录下提供了 ImageNet (inet256), CelebA-HQ (c256) 和 Places2 (p256) 的配置文件，涵盖 "thin", "thick", "every second line", "super-resolution", "expand" 和 "half" 等遮罩类型。你可以像上面的例子一样使用它们。

**如何准备测试数据?**

我们使用 [LaMa](https://github.com/saic-mdal/lama) 进行验证和测试。请遵循他们的说明，并按照配置文件中的指定添加图像。当你使用 `download.sh` 下载数据时，你可以看到我们使用的遮罩示例。

**如何应用于其他图像?**

复制最匹配你数据的配置文件（对于像 CelebA-HQ 那样对齐的人脸使用 `_c256`，对于多样化图像使用 `_inet256`）。然后将 [`gt_path`](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/confs/face_example.yml#L70) 和 [`mask_path`](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/confs/face_example.yml#L71) 设置为你的输入路径。遮罩中，255 表示已知区域，0 表示未知区域（即需要生成的区域）。

**如何应用于其他数据集?**

如果你使用的是人脸、场景或一般图像以外的其他数据，请使用 [guided-diffusion](https://github.com/openai/guided-diffusion) 仓库训练一个模型。请注意，RePaint 是一种推理方案。我们不训练或微调扩散模型，而是根据预训练模型进行调节。

## 代码适配

**如何设计新的调度 (Schedule)?**

在这一[行](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/guided_diffusion/scheduler.py#L180)填入你自己的参数，使用 `python guided_diffusion/scheduler.py` 可视化调度。然后复制一个配置文件，在这些[行](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/confs/face_example.yml#L61-L65)设置你的参数，并使用 `python test.py --conf_path confs/my_schedule.yml` 运行推理。

**如何加速推理?**

以下设置位于配置文件的 [schedule_jump_params](https://github.com/andreas128/RePaint/blob/0fea066b52346c331cdf1bf7aed616c8c8896714/confs/face_example.yml#L61) 键中。你可以如上所述进行可视化。

- 减少 `t_T`，即总步数（不含重采样）。步数越少，每步去除的噪声越多。
- 减少 `jump_n_sample` 以减少重采样次数。
- 通过设置 `start_resampling`，不从一开始就应用重采样，而是从特定时间点开始。

## 代码概览

- **Schedule (调度):** 扩散时间 t 的遍历列表是在这一[行](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L503)获取的。例如 times = [249, 248, 249, 248, 247, 248, 247, 248, 247, 246, ...]
- **Denoise (去噪):** 从 x<sub>t</sub> (较多噪声) 到 x<sub>t-1</sub> (较少噪声) 的反向扩散步骤是在这一[行](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L515)以下完成的。
- **Predict (预测):** 模型在[这里](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L237)被调用，获取 x<sub>t</sub> 和时间 t，预测一个包含 x<sub>t-1</sub> 均值和方差信息的 6 通道张量。然后方差的值范围在[这里](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L252)进行调整。x<sub>t-1</sub> 的均值是通过估计的 [x<sub>0</sub>](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L270) 和 x<sub>t</sub> 的加权和在[这里](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L189)获得的。获得的均值和方差在[这里](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L402)用于采样 x<sub>t-1</sub>。（这是来自 [guided-diffusion](https://github.com/openai/guided-diffusion.git) 的原始反向步骤。）
- **Condition (条件):** 输入图像的已知部分需要具有与扩散模型生成部分相同量的噪声以进行拼接。所需的噪声量在[这里](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L368)计算，并在[这里](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L371)添加到已知部分。生成的部分和采样的部分在[这里](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L373)使用遮罩进行合并。
- **Undo (撤销):** 从 x<sub>t-1</sub> 到 x<sub>t</sub> 的前向扩散步骤是在这一[行](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L536)之后完成的。噪声在[这里](https://github.com/andreas128/RePaint/blob/76cb5b49d3f28715980f6e809c6859b148be9867/guided_diffusion/gaussian_diffusion.py#L176)被添加到 x<sub>t-1</sub>。

## 问题

**你有更多问题吗?**

请打开一个 [issue](https://github.com/andreas128/RePaint/issues)，我们会尝试帮助你。

**你发现了错误吗?**

请创建一个 pull request。例如点击 github 页面右上角的铅笔按钮。

<br>

# RePaint 在多样化内容和缺失区域形状上的表现

蓝色区域是未知的，由 RePaint 填充：

![Denoising Diffusion Probabilistic Models Inpainting](https://user-images.githubusercontent.com/11280511/150803812-a4729ef8-6ad4-46aa-ae99-8c27fbb2ea2e.png)

**注意: RePaint 创造了许多有意义的填充。** <br>

1. **人脸:** 表情和特征，如耳环或痣。<br>
2. **电脑:** 电脑屏幕显示不同的图像、文字，甚至是 logo。<br>
3. **绿植:** RePaint 理解微小的已知部分，并将其融入甲虫、意大利面和植物中。<br>
4. **花园:** 从简单的填充（如窗帘）到复杂的填充（如人）。<br>

<br>

# 极端案例 1: 每隔一行生成

![Denoising_Diffusion_Probabilistic_Models_Inpainting_Every_Second_Line](https://user-images.githubusercontent.com/11280511/150818064-29789cbe-73c7-45de-a955-9fad5fb24c0e.png)

- 输入图像的每隔一行是未知的。
- 大多数 Inpainting 方法在这种遮罩上会失败。

<br>

# 极端案例 2: 图像超分辨率

![Denoising_Diffusion_Probabilistic_Models_Inpainting_Super_Resolution](https://user-images.githubusercontent.com/11280511/150818741-5ed19a0b-1cf8-4f28-9e57-2e4c12303c3e.png)

- Inpainting 只知道步进为 2 的像素。
- 图像的 3/4 需要填充。
- 这相当于使用最近邻核的超分辨率。

<br>

# RePaint 在已知部分上调节扩散模型

- RePaint 使用无条件训练的去噪扩散概率模型。
- 我们在推理过程中根据给定的图像内容进行调节。

![Denoising Diffusion Probabilistic Models Inpainting Method](https://user-images.githubusercontent.com/11280511/180631151-59b6674b-bf2c-4501-8307-03c9f5f593ae.gif)

**单步调节去噪的直觉:**

1. **采样已知部分:** 向图像的已知区域添加高斯噪声。<br> 我们得到一个完全遵循去噪过程的噪声图像。
2. **去噪一步:** 对前一副图像进行一步去噪。这会<br>根据已知区域生成未知区域的内容。
3. **合并:** 合并来自两个步骤的图像。

详情请见第 5 页的算法 1。 [[论文]](https://bit.ly/3b1ABEb)

<br>

# 如何使生成部分与已知部分协调?

- **失败:** 仅使用上述算法时，填充部分与已知部分协调不佳 (n=1)。
- **修复:** 应用 [[4.2 重采样]](https://bit.ly/3b1ABEb) 技术后，图像更加协调 (n>1)。

<img width="1577" alt="Diffusion Model Resampling" src="https://user-images.githubusercontent.com/11280511/150822917-737c00b0-b6bb-439d-a5bf-e73238d30990.png">

<br>

# RePaint 的失败案例

- ImageNet 模型偏向于将内容修复为狗。
- 这是由于 ImageNet 中狗的图片比例很高。

<img width="1653" alt="RePaint Fails" src="https://user-images.githubusercontent.com/11280511/150853163-b965f59c-5ad4-485b-816e-4391e77b5199.png">

<br>

# 用户研究与 SOTA 比较

- 优于基于自回归和基于 GAN 的 SOTA 方法，<br> 除了两个不确定的情况外，对所有遮罩都有 95% 的显著性。
- 用户研究针对三个数据集上的六种不同遮罩进行。
- RePaint 在 44 个案例中的 42 个中优于 SOTA 方法。 [[论文]](https://bit.ly/3b1ABEb)

<br>

# 探索视觉示例

- 数据集: CelebA-HQ, ImageNet, Places2
- 遮罩: 随机线条, 半图, 巨大区域, 稀疏
- 在 [[附录]](https://bit.ly/3b1ABEb) 中探索更多类似的示例。

<img width="1556" alt="Denosing Diffusion Inpainting Examples" src="https://user-images.githubusercontent.com/11280511/150864677-0eb482ae-c114-4b0b-b1e0-9be9574da307.png">

<br>

# 致谢

这项工作得到了 ETH Zürich Fund (OK)、Huawei Technologies Oy (Finland) 项目以及 Nvidia GPU 资助的支持。

本仓库基于 OpenAI 的 [guided-diffusion](https://github.com/openai/guided-diffusion.git)。
