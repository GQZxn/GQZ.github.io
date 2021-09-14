# GQZ.github.io
# HRNet详解

## HRNet：[Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212)[[github](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)]（CVPR2019）这是一篇[state-of-the-art](https://www.stateoftheart.ai/?area=Computer Vision&task=Human Pose Estimation)级别的论文

# 19.9.8 update 本文老板的[HRNet项目主页](https://jingdongwang2017.github.io/Projects/HRNet/index.html)，以及已经应用并取得好成绩的应用方向

| [Pose estimation](https://jingdongwang2017.github.io/Projects/HRNet/PoseEstimation.html) | [Semantic segmentation](https://jingdongwang2017.github.io/Projects/HRNet/SemanticSegmentation.html) | [Face alignment](https://jingdongwang2017.github.io/Projects/HRNet/FaceAlignment.html) | [Image classification](https://jingdongwang2017.github.io/Projects/HRNet/ImageClassification.html) | [Object detection](https://jingdongwang2017.github.io/Projects/HRNet/ObjectDetection.html) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [github](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) | [github](https://github.com/HRNet/HRNet-Semantic-Segmentation) | [github](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) | [github](https://github.com/HRNet/HRNet-Image-Classification) | [github](https://github.com/HRNet/HRNet-Object-Detection)    |

> 本文也可以说是十分硬核的论文，应用在实际任务中效果也是十分给力。**分类网络在视觉识别中占据主导地位**，从图像级分类到区域级分类(目标检测)和像素级分类(语义分割、人体姿态估计和人脸地标检测)。由高到低卷积串联而成的分类网络并不是区域级和像素级分类的好选择，因为它只会导致丰富的低分辨率表示或通过上采样过程得到的低分辨率表示。**本文提出一种高分辨率网络(HRNet)。HRNet通过并行连接高分辨率到低分辨率卷积来保持高分辨率表示，并通过重复跨并行卷积执行多尺度融合来增强高分辨率表示。在像素级分类、区域级分类和图像级分类中，证明了这些方法的有效性。**

------

（消息来源[刷新三项COCO纪录！姿态估计模型HRNet开源了，中科大微软出品 | CVPR](https://mp.weixin.qq.com/s/_fSouDdv6L9zfAchQWx1GA)）

中科大和微软亚洲研究院，发布了新的人体**姿态估计模型**，刷新了三项COCO纪录，还中选了**CVPR 2019**。

这个名叫**HRNet**的神经网络，拥有与众不同的并联结构，可以随时**保持高分辨率表征**，不只靠从低分辨率表征里，恢复高分辨率表征。如此一来，姿势识别的效果明显提升：

在COCO数据集的**关键点检测**、**姿态估计**、**多人姿态估计**这三项任务里，HRNet都超越了所有前辈。

![img](https://img-blog.csdnimg.cn/20190908150909441.png)

------

# 一、Abstract摘要&Introduction介绍

## Abstract

​       在这篇论文中，我们主要研究**人的姿态问题**(human pose estimation problem)，着重于输出**可靠的高分辨率表征**(reliable highresolution representations)。现有的大多数方法都是从高分辨率到低分辨率网络(high-to-low resolution network)产生的低分辨率表征中恢复高分辨率表征。相反，我们提出的网络能在整个过程中都保持**高分辨率的表征**。

​       我们从高分辨率子网络(high-resolution subnetwork)作为第一阶段开始，逐步增加高分辨率到低分辨率的子网，形成更多的阶段，**并将多分辨率子网并行连接**。我们进行了多次多尺度融合multi-scale fusions，使得每一个高分辨率到低分辨率的**表征**都从其他并行表示中反复接收信息，从而得到丰富的高分辨率表征。因此，预测的**关键点热图**可能更准确，在空间上也更精确。通过 ***COCO keypoint detection\*** 数据集和 ***MPII Human Pose\*** 数据集这两个基准数据集的pose estimation results，我们证明了网络的有效性。此外，我们还展示了网络在 ***Pose Track*** 数据集上的姿态跟踪的优越性。

 

## Introduction

​       二维人体姿态估计 ***2D human pose\*** 是计算机视觉中一个基本而又具有挑战性的问题。目标是定位人体的解剖关键点(如肘部、腕部等)或部位。它有很多应用，包括**人体动作识别、人机交互、动画**(***human action recognition, human-computer interaction, animation\***)等。本文着力于研究**single-person pose estimation**，这是其他相关问题的基础，如***multiperson pose estimation***[6,27,33,39,47,57,41,46,17,71]，***video pose estimation and tracking***[49,72]等。（引用的论文见博客最后，许多论文还是值得一读的）

   最近的发展表明，深度卷积神经网络已经取得了最先进的性能。大多数现有的方法通过一个网络(通常由高分辨率到低分辨率的子网串联而成)传递输入，然后提高分辨率。例如，**Hourglass**[40]通过**对称的低到高分辨率**(symmetric low-to-high process)过程恢复高分辨率。**SimpleBaseline**[72]采用少量的**转置卷积层**(***transposed convolution layers***)来生成高分辨率的表示。此外，***dilated convolutions***还被用于放大高分辨率到低分辨率网络(high-to-low resolution network)的后几层[27,77](如VGGNet或ResNet)。

​       我们提出了一种新的**架构，即高分辨率网络(HRNet)**，它能够在整个过程中维护高分辨率的表示。我们从高分辨率子网作为第一阶段始，**逐步增加高分辨率到低分辨率的子网**(gradually add high-to-low resolution subnetworks)，形成更多的阶段，并将**多分辨率子网并行连接。**在整个过程中，我们通过在**并行的多分辨率子网络**上反复交换信息来进行多尺度的重复融合。我们通过网络输出的高分辨率表示来估计关键点。生成的网络如图所示。

![img](https://img-blog.csdnimg.cn/20190228210015456.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

与现有的广泛用于姿态估计(pose estimation)的网络[40,27,77,72]相比，我们的网络有两个好处。

- (i)我们的方法是**并行连接高分辨率到低分辨率的子网**，而不是像大多数现有解决方案那样**串行连接**。因此，我们的方法能够保持高分辨率，而不是通过一个低到高的过程恢复分辨率，因此预测的热图可能在空间上更精确。***parallel high-to-low resolution subnetworks***
- (ii)大多数现有的融合方案都将低层和高层的表示集合起来。相反，我们使用**重复的多尺度融合**，利用相同深度和相似级别的低分辨率表示来提高高分辨率表示，反之亦然，从而使得高分辨率表示对于姿态的估计也很充分。因此，我们预测的热图可能更准确。***multi-resolution subnetworks (multi-scale fusion)***

   我们通过实验证明了在两个基准数据集(Benchmark): COCO关键点检测数据集(***keypoint detection***)[36]和MPII人体姿态数据集(***Human Pose dataset***)[2]上优越的关键点检测性能。此外，我们还展示了我们的网络在PoseTrack数据集上视频姿态跟踪的优势[1]

 

# 二、相关研究

​       传统的单人体姿态估计大多采用**概率图**形模型或图形结构模型[79,50]，近年来通过深度学习对一元能量和对态能量unary and pair-wise energies进行更好的建模[9,65,45]或模仿迭代推理过程[13]，对该模型进行了改进。

​       目前，**深度卷积神经网络**提供了主流的解决方案[20,35,62,42,43,48,58,16]。主要有两种方法：**回归关键点位置regressing the position of keypoints[66,7]和估算关键点热图estimating keypoint heatmaps[13,14,78]，然后选择热值最高的位置作为关键点。**

​       大多数卷积神经网络估计**对于关键点的热图是由一个stem子网网络**类似于分类网络，这**降低了分辨率，把相同的分辨率作为输入的一个主要原因，后跟一个回归量估算的热图关键点位置估计，然后转变为完整的决议。**主体主要采用高到低和低到高分辨率的框架，可能增加多尺度融合multi-scale fusion和深度监督学习 and intermediate (deep) supervision。

### High-to-low and low-to-high.

   high-to-low process的目标是生成低分辨率和高分辨率的表示，low-to-high process的目标是生成高分辨率的表示[4,11,23,72,40,62]。这两个过程可能会重复多次，以提高性能[77,40,14]。

   具有代表性的网络设计模式包括：

   (i)Symmetric high-to-low and low-to-high processes。Hourglass及其后续论文[40,14,77,31]将low-to-high proces设计为high-to-low process的镜子。

   (ii)Heavy high-to-low and light low-to-high。high-to-low process是基于ImageNet分类网络，如[11,72]中使用的ResNet，low-to-high process是简单的几个双线性上采样[11]或转置卷积[72]层。

   (iii)Combination with dilated convolutions。在[27,51,35]中，ResNet或VGGNet在最后两个阶段都采用了扩张性卷积来消除空间分辨率的损失，然后采用由light lowto-high process来进一步提高分辨率，避免了仅使用dilated convolutions的昂贵的计算成本[11,27,51]。图2描述了四种具有代表性的姿态估计网络。

![img](https://img-blog.csdnimg.cn/20190228205948680.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

### Multi-scale fusion.

   最直接的方法是将多分辨率图像分别送入多个网络，并聚合输出响应映射[64]。Hourglass[40]及其扩展[77,31]通过跳过连接，将high-to-low process中的**低级别特征**逐步组合为low-to-high process中的相同分辨率的高级别特性。在cascaded pyramid network[11]中，globalnet将high-to-low process中的低到高级别特征low-to-high level feature逐步组合到low-to-high process中，refinenet将通过卷积处理的低到高特征进行组合。我们的方法重复多尺度融合，部分灵感来自深度融合及其扩展[67,73,59,80,82]。

### Intermediate supervision.

   图像分类早期开发的中间监督或深度监督[34,61]，也用于帮助深度网络训练和提高热图估计质量，如[69,40,64,3,11]。Hourglass[40]和卷积人体姿态方法[69]将中间热图作为剩余子网络的输入或输入的一部分进行处理。

### Our approach.

   我们的网络并行地连接高到低的子网。它保持了高分辨率的表示，通过整个过程的空间精确热图估计。它通过重复融合高到低子网产生的高分辨率表示来生成可靠的高分辨率表示。我们的方法不同于大多数现有的工作，它们需要一个独立的从低到高的上采样过程，并聚合低级和高级表示。该方法在不使用中间热图监控的情况下，具有较好的关键点检测精度和计算复杂度及参数效率。

   有相关的多尺度网络进行分类和分割[5,8,74,81,30,76,55,56,24,83,55,52,18]。我们的工作在一定程度上受到了其中一些问题的启发[56,24,83,55]，而且存在明显的差异，使得它们不适用于我们的问题。Convolutional neural fabrics[56]和interlinked CNN[83]由于缺乏对每个子网络(depth, batch)的合理设计和多尺度融合，分割结果的质量不高。grid network[18]是多个权重共享U-Nets组合，由两个多分辨率表示的独立融合过程组成；第一阶段，信息仅从高分辨率发送到低分辨率；在第二阶段，信息只从低分辨率发送到高分辨率，因此竞争力较低。Multi-scale densenets[24]没有目标，无法生成可靠的高分辨率表示。

 

# 三、Approach

   人体姿态估计（Human pose estimation），即关键点检测，旨在检测K个部分关键点的位置(例如,手肘、手腕等)从一个图像的大小W×H×3。stateof-the-art级别的方法是将这个问题转变为估计K热图的大小![img](https://img-blog.csdnimg.cn/20190228205501728.png)，其中每个热图 Hk 表示第k个关键点的位置置信度。

   我们遵循广泛适用的pipeline[40, 72, 11]使用卷积网络预测human keypoint，这是由stem组成的两个步长卷积网络降低分辨率，主体输出特性映射具有相同的分辨率作为它的输入特征图，和一个回归量估算的热图关键点位置选择和转换为完整的分辨率。我们主要关注主体的设计，并介绍我们的高分辨率网络(HRNet)，如图1所示。

 

### Sequential multi-resolution subnetworks

   现有的姿态估计网络是将高分辨率到低分辨率的子网络串联起来，每个子网络形成一个阶段，由一系列卷积组成，相邻子网络之间存在一个下采样层，将分辨率减半。

![img](https://img-blog.csdnimg.cn/20190228205912688.png)

 

### Parallel multi-resolution subnetworks

   我们以高分辨率子网为第一阶段，逐步增加高分辨率到低分辨率的子网，形成新的阶段，并将多分辨率子网并行连接。因此，后一阶段并行子网的分辨率由前一阶段的分辨率和下一阶段的分辨率组成。一个包含4个并行子网的网络结构示例如下：

![img](https://img-blog.csdnimg.cn/20190228210116101.png)

 

### Repeated multi-scale fusion

   我们引入了跨并行子网的交换单元，使每个子网重复接收来自其他并行子网的信息。下面是一个展示信息交换方案的示例。我们将第三阶段划分为若干个交换块，每个块由3个并行卷积单元与1个交换单元跨并行单元进行卷积，得到：

![img](https://img-blog.csdnimg.cn/20190228210214110.png)

我们将在图3中说明交换单元，并在下文中给出公式。为了便于讨论，我们去掉下标s和上标b。

![img](https://img-blog.csdnimg.cn/20190228210344793.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

   输入为s响应映射：![img](https://img-blog.csdnimg.cn/2019022821042874.png)。输出为s响应图：![img](https://img-blog.csdnimg.cn/20190228210440270.png)，其分辨率和宽度与输入相同。每个输出都是输入映射的集合![img](https://img-blog.csdnimg.cn/20190228211548483.png)。各阶段的交换单元有额外的输出图![img](https://img-blog.csdnimg.cn/20190228211600799.png)。

   函数![img](https://img-blog.csdnimg.cn/20190228211632912.png))从分辨率i到k对![img](https://img-blog.csdnimg.cn/20190228211740879.png)上采样或下采样组成。我们采用步长为3×3的卷积做下采样。例如，向一个步长为3×3卷积做步长为2x2的下采样。两个连续的步长为3×3的卷积使用步长为2的4被上采样。对于上采样，我们采用最简单的最近邻抽样，从1×1卷积校准通道的数量。如果i = k，则a(.,.)只是一个识别连接：![img](https://img-blog.csdnimg.cn/20190228212025150.png)。

 

### Heatmap estimation

   我们将热图简单地从上一个交换单元输出的高分辨率表示进行回归，这在经验上是很有效的。损失函数定义为均方误差，用于比较预测的热图和groundtruth heatmpas。groundtruth heatmpas是采用二维高斯分布，以每个关键点的grouptruth位置为中心，标准差为1像素生成的。

 

### Network instantiation

   根据ResNet的设计规则，我们实例化了用于关键点热图估计的网络，将深度分布到每个阶段，并将通道数分布到每个分辨率。

   我们的HRNet包含四个阶段，主体为四个并行的子网，其分辨率逐渐降低到一半，相应的宽度(通道数)增加了一倍。第一阶段包含4个剩余单位，每个单位都和ResNet-50一样，是由一个宽度为64的bottleneck组成，紧随其后的是一个3x3卷积特征图的宽度减少到（C），第二，第三，第四阶段分别包含1、4、3个交换块。一个交换块包含4个剩余单元，其中每个单元在每个分辨率中包含2个3x3的卷积，以及一个分辨率的交换单元。综上所述，共有8个交换单元，即共进行8次多尺度融合。

   在我们的实验中，我们研究了一个小网和一个大网：HRNet-W32和HRNet-W48，其中32和48分别代表最后三个阶段高分辨率子网的宽度（C）。其他三个并行子网的宽度为64，128，256的HRNet-W32，以及HRNet-W48：96，192，384。

 

# 四、实验

## 1. COCO Keypoint Detection

**Dataset.** COCO数据集[36]包含200000幅图像和250000个带有17个关键点的person实例。我们在COCO train2017数据集上训练我们的模型，包括57K图像和150K person实例。我们在val2017和test-dev2017上评估我们的方法，这两个集合分别包含5000幅图像和20K幅图像。

**Evaluation metric. 这里的[标准](http://cocodataset.org/#keypoints-eval)使用过COCO的应该知道。**标准的评价指标是基于对象关键点相似(Object Keypoint Similarity (OKS)): ![OKS = \frac{\sum_{i} \exp \left(-d_{i}^{2} / 2 s^{2} k_{i}^{2}\right) \delta\left(v_{i}>0\right)}{\sum_{i} \delta\left(v_{i}>0\right)}](https://private.codecogs.com/gif.latex?OKS%20%3D%20%5Cfrac%7B%5Csum_%7Bi%7D%20%5Cexp%20%5Cleft%28-d_%7Bi%7D%5E%7B2%7D%20/%202%20s%5E%7B2%7D%20k_%7Bi%7D%5E%7B2%7D%5Cright%29%20%5Cdelta%5Cleft%28v_%7Bi%7D%3E0%5Cright%29%7D%7B%5Csum_%7Bi%7D%20%5Cdelta%5Cleft%28v_%7Bi%7D%3E0%5Cright%29%7D) 。这里di是检测到的关键点与对应的地面真相之间的欧式距离，vi是地面真相的可见性标志，s是对象尺度，ki是控制衰减的每个关键点常量。我们报告的标准平均精度和召回分数:![AP^{50 }](https://private.codecogs.com/gif.latex?AP%5E%7B50%20%7D) (AP at OKS = 0.50) AP75, AP (AP得分的平均值在10个位置，OKS = 0.50, 0.55，…,0.90,0.95;中等对象的![AP^M](https://private.codecogs.com/gif.latex?AP%5EM)，![AP^L](https://private.codecogs.com/gif.latex?AP%5EL)为大对象，以及当OKS = 0.50, 0.55,…,0.90,0.955时的AR。

**Training.** 我们将人体检测盒的高宽比扩展到固定的长宽比:高:宽= 4:3，然后从图像中裁剪出盒子，调整为固定的大小，256×192或384×288。数据增加包括随机旋转(- 45◦,45◦)，随机的规模([0.65, 1.35])，然后翻转。在[68]之后，一半的数据增强。我们使用Adam优化器[32]。学习进度按照设定[72]。基本学习率设置为1e - 3，在第170和200epochs训练，在210个epochs内结束。

**Testing.**类似于[47，使用:使用person检测器检测person实例，然后预测检测关键点。我们使用SimpleBaseline2提供的同一个人检测器
[72]用于验证集和测试开发集。根据通常的做法[72,40,11]，我们通过对原始图像和翻转图像的头部图进行平均来计算热图。每个关键点的位置是通过调整最高热值的位置，并在从最高响应到第二高响应的方向上偏移四分之一来预测的。

**Results on the validation set.** 我们在表1中报告了我们的方法和其他最先进方法的结果。我们的小型网络HRNet-W32，从零开始训练，输入大小为256192，获得了73.4分的AP分数，优于其他相同输入大小的方法。(i)与Hourglass[40]相比，我们的小网络提高了AP的人位置估计。我们的网络的GFLOPs要低得多，不到一半，而参数的数量是相似的，我们的略大一些。(ii)与CPN [11] w/o和w/ OHKM相比，我们的网络模型尺寸略大，复杂度略高，分别获得4.8和4.0分的增益。(3)相比best-performed SimpleBaseline[72],我们小净HRNet-W32获得显著改善:3.0分获得的骨干与类似的型号大小和GFLOPs ResNet-50,和1.4分获得的骨干ResNet152模型尺寸(# Params)和难吃的东西是我们的两倍。

我们的网络可以从(i)针对ImageNet分类问题的模型预训练中获益:HRNet-W32的增益为1.0分;(ii)通过增加宽度来增加容量:我们的大net HRNet-W48在输入大小分别为256 192和384 288的情况下得到了0.7和0.5的改进。

考虑到输入大小为384 288，我们的HRNet-W32和HRNet-W48得到了75.8和76.3 AP，与输入大小为256 192相比，它们分别有1.4和1.2的改进。与以ResNet-152为骨干的SimpleBaseline[72]相比，我们的HRNet-W32和HRNetW48在AP方面分别获得1.5和2.0分，计算成本分别为45%和92.4%。

**Results on the test-dev set.** 表2报告了我们的方法的姿态估计性能和现有的最先进的方法。我们的方法明显优于自底向上方法。另一方面，我们的小型网络HRNet-W32的AP达到了74.9。它优于所有其他自顶向下的方法，并且在模型大小(#Params)和计算复杂度(GFLOPs)方面更有效。我们的大模型HRNet-W48获得了最高的75.5 AP，与相同输入大小的SimpleBaseline[72]相比，我们的小网络和大网络分别得到了1.2和1.8的改进。使用来自AI Challenger[70]的训练数据，我们的单一大网络可以获得77.0的AP。

### 验证集结果对比： 

![img](https://img-blog.csdnimg.cn/20190228212828505.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

### 测试集结果对比：

![img](https://img-blog.csdnimg.cn/20190228212931116.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

##  

## 2. MPII Human Pose Estimation

**Dataset.** MPII人体姿态数据集[2]由来自具有全身姿态注释的广泛现实活动的图像组成。大约有25 k图像40 k,哪里有12 k受试者进行测试和剩下的训练集的主题。数据扩充和训练策略是相同的COCO，除了输入大小裁剪256 256公平与其他方法进行比较。

**Testing.** 测试过程与COCO中的测试过程几乎相同，只是我们采用了标准的测试策略，使用提供的人员框而不是检测到的人员框。在[14,77,62]之后，进行了六尺度金字塔测试程序。

**Evaluation metric.** 使用标准度量[2]，即PCKh(正确关键点的头部归一化概率)得分。联合是正确的如果它属于αl groundtruth位置的像素,其中α是一个常数和l是头的大小对应于真实的对角线长度的60%头边界框。PCKh@0.5(α= 0.5)评分。

**Results on the test set.** 表3和表4显示了PCKh@0.5的结果、模型大小和最常用方法的GFLOPs。我们使用ResNet-152作为输入大小为256 256的主干，重新实现了SimpleBaseline[72]。我们的HRNet-W32实现了92.3 PKCh@0.5的评分，并优于堆叠沙漏方法[40]及其扩展[58、14、77、31、62]。我们的结果与之前发表在20183年11月16日排行榜上的结果中最好的结果一致[62]。我们想指出的是，该方法[62]是对我们的方法的补充，利用成分模型来学习人体的构型，采用多层次的中间监督，我们的方法也可以从中受益。我们还测试了我们的大网络HRNetW48，得到了相同的结果92.3。原因可能是这个数据集中的性能趋于饱和。

注意：

- 使用翻转测试
- 输入大小为 256x256
- 我们之前的工作： [*Simple Baselines for Human Pose Estimation and Tracking*](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html)

***结果对比***

![img](https://img-blog.csdnimg.cn/20190228213008423.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

 

![img](https://img-blog.csdnimg.cn/20190908154358368.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

## 3. Application to Pose Tracking

**Dataset.** PoseTrack[28]是视频中用于人体姿态估计和关节跟踪的大规模基准。该数据集基于流行的MPII人体姿态数据集提供的原始视频，包含550个视频序列和66374帧。视频序列被分成用于训练、验证和测试的视频分别为292,50,208个。训练视频的长度在41 151帧之间，从视频中心到30帧之间都有密集的注释。验证/测试视频的帧数在65 298帧之间。MPII Pose数据集中的关键帧周围的30帧被密集地注释，然后每四帧被注释一次。总的来说，这大约包括23000个带标签的帧和153615个摆姿势的注释。

**Evaluation metric.** 我们从两方面对结果进行了评估:帧间多姿态估计和多姿态跟踪。姿态估计是通过平均平均精度(mAP)来评估的，如[51,28]。多目标跟踪精度(MOTA)是多目标姿态跟踪的评价指标[38,28]。详情见[28]。

**Training.** 我们在PoseTrack2017训练集上训练HRNet-W48进行单人姿态估计，其中网络由COCO数据集上预训练的模型初始化。我们将person框作为网络的输入从训练帧中的带注释的关键点中提取出来，方法是将所有关键点(对于一个人)的边界框扩展15%。包括数据扩充在内的训练设置与COCO基本相同，只是学习进度不同(现在是微调):学习速度从1e4开始，第10历元下降到1e5，第15历元下降到1e6;迭代在20个纪元内结束。

**Testing.** 我们跟随[72]来跟踪帧间的姿势。它由三个步骤组成:人体姿态检测与传播、人体姿态估计和姿态关联跨帧。我们使用与SimpleBaseline相同的person box检测器[72]，根据FlowNet 2.0[26] 4计算的光流传播预测的关键点，将检测到的box传播到附近的帧中，然后对box去除进行非最大抑制。姿态关联方案是基于一帧内的关键点与根据光流从邻近帧传播的关键点之间的目标关键点相似性。然后利用贪心匹配算法计算关键字之间的对应关系附近的帧。更多细节见[72]。

![img](https://img-blog.csdnimg.cn/20190908154415530.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

**Results on the PoseTrack2017 test set** 表5报告了结果。我们的大网络- HRNet-W48取得了更好的成绩，74.9的地图评分和57.9的MOTA评分。与第二好的方法SimpleBaseline中以ResNet-152为骨干的FlowTrack[72]相比，我们的方法在mAP和MOTA方面分别获得了0.3和0.1分的增益。相对于FlowTrack[72]的优势与COCO关键点检测和MPII人体姿态估计数据集的优势是一致的。这进一步说明了我们的姿态估计网络的有效性。表6给出的COCO验证集表明，多尺度融合是有益的，融合越多，性能越好。

**Resolution maintenance.** 我们研究了HRNet的一种变体的性能:所有四个高分辨率到低分辨率的子网都是在开始时添加的，而且深度相同;融合方案与我们的相同。我们的HRNet-W32和变体(类似的#Params和GFLOPs)都是从零开始训练，并在COCO验证集上进行测试。变体的AP为72.5，低于我们的小型net HRNet-W32的73.4 AP。我们认为，其原因是在低分辨率子网的早期阶段提取的低层特征没有那么有用。此外，在没有低分辨率并行子网的情况下，参数和计算复杂度相似的简单高分辨率网络的性能要低得多。

**Representation resolution.** 研究了表示分辨率对姿态估计性能的影响，主要从两个方面进行了研究:从高分辨率的特征图出发，对热图的质量进行检测并研究了输入大小对质量的影响。

我们训练我们的小的和大的网络初始化的模型预训练的ImageNet分类。我们的网络从高到低输出四个响应映射。低分辨率响应图的热图预测质量过低，AP评分低于10分。图5报告了AP在其他三个地图上的得分。对比表明，分辨率对关键点预测质量有一定的影响

。图6显示了与SimpleBaseline (ResNet50)相比，输入图像大小如何影响性能[72]。我们可以发现，较小的输入大小的改进比较大的输入大小的改进更为显著，例如，256 x 192的改进为4.0分，128 x 96的改进为6.3分。原因是我们始终保持高分辨率。这意味着我们的方法在实际应用中更有优势，计算成本也是一个重要因素。另一方面，输入大小为256192的方法优于输入大小为384 x 288的SimpleBaseline[72]。

## 4. Ablation Study

我们研究了该方法中每个组件对COCO关键点检测数据集的影响。除对输入尺寸影响的研究外，所有结果均在256×192的输入尺寸下得到。

**Repeated multi-scale fusion.** 对重复多尺度融合的效果进行了实证分析。我们研究网络的三种变体。(a) W/o中间交换单元(1融合):除最后一个交换单元外，多分辨率子网之间没有交换。(b) W/跨级交换单元(3个融合):每个阶段内并行子网之间没有交换。(c)跨级和级内交换单元(共8个融合):这是我们提出的方法。所有的网络都是从零开始训练的。有关

![img](https://img-blog.csdnimg.cn/201909081544296.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

#  

# 五、Conclusion and Future Works

   在本文中，我们提出了一个高分辨率的网络，用于人体姿态估计，产生准确和空间精确的关键点热图。成功的原因有两个方面：(1)全程保持高分辨率，不需要恢复高分辨率；(2)多次融合多分辨率表示，呈现可靠的高分辨率表示。

   未来的工作包括应用于其他密集预测任务，如语义分割、图标检测、人脸对齐、图像翻译，以及研究以一种不那么轻量的方式聚合多分辨率表示。

# Appendix 

Results on the MPII Validation Set Table7

我们提供了关于MPII验证集[2]的结果。我们的模型在MPII训练集的子集上进行训练，并在2975幅图像的heldout留出验证集上进行评估。训练过程与整个MPII训练集的训练过程相同，热图计算为原始图像和翻转图像热图的平均值，用于测试。在[77,62]之后，我们还进行了六尺度金字塔测试six-scale pyramid testing procedure(多尺度测试)。结果如表7所示。

![img](https://img-blog.csdnimg.cn/20190908154937844.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

Results on the PoseTrack Dataset Table8\9

我们提供了PoseTrack数据集[1]上所有关键点的结果。表8显示了PoseTrack2017数据集中的多人姿态估计性能。

![img](https://img-blog.csdnimg.cn/20190908154944569.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

我们的HRNet-W48在验证集和测试集上的mAP分别达到77.3和74.9点，比之前的最先进的方法[72]分别高出0.6和0.3点。我们在PoseTrack2017测试集中提供了更详细的多人姿态跟踪性能结果，作为本文报告结果的补充，如表9所示。

![img](https://img-blog.csdnimg.cn/20190908154949936.png)

 

# 六、References

![img](https://img-blog.csdnimg.cn/20190228201734432.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20190228201750946.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

 

![img](https://img-blog.csdnimg.cn/20190228201800161.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

