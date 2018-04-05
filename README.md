# Pose-IOS

这是基于[Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)开发的移动端(ios)人体关键点检测程序.

[Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1611.08050.pdf)这篇文章的出现对pose界产生的很大影响，多人识别，且精度很高。

但是由于它所采用的网络结构层级多，参数量大，运算量大，所以很难直接在移动端进行应用。
因此所做的是探索在保证该算法精度下降范围允许的情况下来减小模型计算量。

主要采用的方法是用[MobileNet](https://arxiv.org/abs/1704.04861.pdf)中的Seperable Convolution 结构来替换原来计算量很大的卷积结构。最终实现的效果在IOS端能达到每秒10帧。

![image](https://github.com/yukang2017/Pose-IOS/blob/master/pose_by_mobile.gif)
