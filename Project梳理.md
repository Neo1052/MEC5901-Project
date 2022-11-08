# 任务
![f9635f8289eecc2f11631734c7901d4.jpg](https://cdn.nlark.com/yuque/0/2022/jpeg/26322346/1667914696794-1b1afce3-1390-4be1-8cdb-3c478fa04be6.jpeg#averageHue=%23e3e1db&clientId=ubf821477-4037-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=380&id=u773df6ea&margin=%5Bobject%20Object%5D&name=f9635f8289eecc2f11631734c7901d4.jpg&originHeight=475&originWidth=770&originalType=binary&ratio=1&rotation=0&showTitle=false&size=68580&status=done&style=none&taskId=u6ef9e82b-2fa0-4edb-a02d-2379406dcb8&title=&width=616)

# 问题背景：
（1）频分双工模式下需要将下行链路的CSI反馈回基站，以便对数据流进行预编码。
![](https://cdn.nlark.com/yuque/0/2022/png/26322346/1667892694204-49cb75fb-58f3-4836-a495-fcae8bacb377.png#averageHue=%23fbfbfb&crop=0&crop=0&crop=1&crop=1&from=url&id=iG8Qf&margin=%5Bobject%20Object%5D&originHeight=584&originWidth=1192&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
（2）直接传输会消耗过多资源，通常使用矢量量化或基于码本的方法来减少反馈开销。
（3）传统的CS（压缩感知）方法有以下三个缺点：假设了信道稀疏、使用随机投影而没用充分利用信道结构、使用迭代方法重建而速度较慢。
（4）论文使用了深度学习方案解决了上述问题：CsiNet 感知与恢复网络

- 编码器：CsiNet不是使用随机投影，而是从原始信道矩阵学习变换，通过训练数据压缩表示（码字）。该算法与人类对频道分布的知识无关，而是直接从训练数据中学习如何有效地使用频道结构。
- 解码器：CsiNet学习从码字到原始信道的逆变换。逆变换是非迭代的，并且比迭代算法快多个数量级。

（5）该方法可以在FDD MIMO系统中用作反馈协议。事实上，CsiNet与DL中的自动编码器[9，Ch.14]密切相关，后者用于学习一组数据的表示（编码），通常用于降维。
（6）使用该方案可以以显著提高的重建质量来恢复CSI。即使以过低的压缩率进行重构，也保留了允许有效波束形成增益的足够内容。

# 系统模型与CSI反馈——设计方案
![image.png](https://cdn.nlark.com/yuque/0/2022/png/26322346/1667915271833-bc733416-ca30-46ab-ad00-65fa044cb944.png#averageHue=%23f7f6f6&clientId=ub1414a0a-d018-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=295&id=u6c0e01e3&margin=%5Bobject%20Object%5D&name=image.png&originHeight=369&originWidth=1255&originalType=binary&ratio=1&rotation=0&showTitle=false&size=104333&status=done&style=none&taskId=u86b5282f-f491-4132-9160-00b0d6b8ba0&title=&width=1004)
（1）考虑这样一个系统：Nt个发射天线 + 单个接收用户 （整个运行的流程是怎么样的？）
![](https://cdn.nlark.com/yuque/0/2022/png/26322346/1667895140689-45f607eb-1af6-42b4-88f3-3082bd8eee0b.png#averageHue=%23fcfcfc&crop=0&crop=0&crop=1&crop=1&from=url&id=nCXTm&margin=%5Bobject%20Object%5D&originHeight=667&originWidth=1109&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
（2）第n个子载波处的接收信号如下
![image.png](https://cdn.nlark.com/yuque/0/2022/png/26322346/1667915655685-04ebacbb-9bf7-45d6-a928-ac6490692a68.png#averageHue=%23fafafa&clientId=uba954218-1460-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=33&id=ue2f9c7d6&margin=%5Bobject%20Object%5D&name=image.png&originHeight=41&originWidth=163&originalType=binary&ratio=1&rotation=0&showTitle=false&size=2916&status=done&style=none&taskId=u7a2637cc-c4e1-4faf-a6c8-4d981d559ec&title=&width=130.4)
其中分别是：第n个子载波的信道矢量、预编码矢量、数据承载符号和附加噪声
（3）H~ ：CSI （所有子载波组合）
（4）为了减少反馈开销，我们提出可以使用2D离散傅里叶变换（DFT）在角延迟域中稀疏~H
![image.png](https://cdn.nlark.com/yuque/0/2022/png/26322346/1667915784150-95eb508e-f754-402a-b777-4131e5fc7564.png#averageHue=%23f7f7f7&clientId=uba954218-1460-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=31&id=u4deea71d&margin=%5Bobject%20Object%5D&name=image.png&originHeight=39&originWidth=124&originalType=binary&ratio=1&rotation=0&showTitle=false&size=2283&status=done&style=none&taskId=u4b712a54-d2be-4985-8ae5-d2c6d437146&title=&width=99.2)   
（5）COST 2100 信道模型【发挥了什么作用？】
![image.png](https://cdn.nlark.com/yuque/0/2022/png/26322346/1667915837448-15158117-44a0-43fd-b9dc-da09b1d530d9.png#averageHue=%23f4f3f2&clientId=uba954218-1460-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=233&id=u9e1f0c91&margin=%5Bobject%20Object%5D&name=image.png&originHeight=291&originWidth=440&originalType=binary&ratio=1&rotation=0&showTitle=false&size=40035&status=done&style=none&taskId=uf94e40f1-dc28-4e94-8504-404c827d91d&title=&width=352)   
（6）使用具有半波长间隔的均匀线性阵列（ULA）进行参数化【波束成形相关？】   
（7）在延迟域中，只有H的前Nc行包含值，因为多径到达之间的时间延迟位于有限的周期内。因此，我们可以保留H的前Nc行并删除其余行。【这一条要如何解释？缺乏相关的知识，有点不太清楚】  
（8）编码器：   
![image.png](https://cdn.nlark.com/yuque/0/2022/png/26322346/1667916109071-1f06ede4-e978-4ff7-aff2-ac5c6975e108.png#averageHue=%23fafafa&clientId=uba954218-1460-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=31&id=u88245d1e&margin=%5Bobject%20Object%5D&name=image.png&originHeight=39&originWidth=129&originalType=binary&ratio=1&rotation=0&showTitle=false&size=2099&status=done&style=none&taskId=ud24004e0-38e1-4a7e-a62b-e57dd125e2c&title=&width=103.2)
可以将信道矩阵变换为M维向量；需要设计相应的逆变换器   

CSI反馈方法如下。一旦在UE侧获取了信道矩阵~H，我们在（2）中执行2D DFT以获得截断矩阵H，然后使用编码器（3）生成码字s。返回给BS，BS使用解码器（4）获得H。可以通过执行逆DFT来获得空间频域中的最终信道矩阵。   

# CSINET
![image.png](https://cdn.nlark.com/yuque/0/2022/png/26322346/1667916415133-37171132-0c00-4116-8955-33ee3a434b6b.png#averageHue=%23eef1e8&clientId=uba954218-1460-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=307&id=u54649441&margin=%5Bobject%20Object%5D&name=image.png&originHeight=384&originWidth=1219&originalType=binary&ratio=1&rotation=0&showTitle=false&size=114959&status=done&style=none&taskId=u0b161bb8-b8e3-46ba-b2fd-0d831c49b27&title=&width=975.2)   
（1）利用传统神经网络（CNN）作为编码器和解码器   
（2）编码器：卷积层——重塑层——全连接层——生成码字 【前两层模拟CS的投影并用作编码器】   
（3）解码器：解码——重塑层——RefineNet 细化单元【输入层 - 卷积1 - 卷积2 - 卷积3（重构）】——填充——激活函数——细化.....   
（4）特点   
RefineNet装置的两个特点如下。首先，RefineNet单元的输出大小等于信道矩阵大小。这一概念受到[10]和[11]的启发。
为了降低维数，几乎所有CNN的传统实现都涉及池化层，这是一种下采样形式。与传统实现相比，我们的目标是细化而不是降维。【如何做到细化？】
其次，在RefineNet单元中，我们引入了身份快捷连接，直接将数据流传递到后面的层。这种方法受到深度残差网络[12]，[17]的启发，它避免了由多个叠加非线性变换引起的消失梯度问题。   
（5）实验结论
实验表明，两个RefineNet单元产生了良好的性能。添加更多的RefineNet单元不会显著提高重建质量，但会增加计算复杂性。一旦信道矩阵被一系列RefineNet单元细化，信道矩阵就被输入到最终的卷积层，并且使用S形函数将值缩放到[0，1]范围。通过增加特征图（即S3）的数量，可以扩展CsiNet以处理涉及UE处多个天线的情况。我们将利用UE天线之间的空间相关性作为未来研究的主题。

# 训练与实验方法
还没有细看，但有几个问题：
（1）学习的过程是监督还是非监督？任务是回归、聚类or？
（2）是否需要考虑多个机器学习方案？要怎么说明我们选择CNN的原因？
（3）每一层的设计，以及理由？【感觉这个更加关键，前面的部分可以在理解的基础上用作者的代码实现，这一部分需要我们拓展】

# 现在的实施思路
阶段1：搞清论文中的细节，提出研究内容的完整框架，
阶段2：
阶段3：
![](https://cdn.nlark.com/yuque/0/2022/jpeg/26322346/1667914696794-1b1afce3-1390-4be1-8cdb-3c478fa04be6.jpeg#averageHue=%23e3e1db&crop=0&crop=0&crop=1&crop=1&from=url&height=384&id=jDEvk&margin=%5Bobject%20Object%5D&originHeight=475&originWidth=770&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=623)

- [ ] 写出所研究问题的背景，对存在的问题进行描述
- [ ] 将论文当中的具体问题提出来进行描述，并从头到尾完整地以数学公式进行表达（直接在论文中梳理即可），明确一下从机器学习的角度，这是一个什么问题（回归、聚类。。。。）
- [ ] 跑一下代码，把论文中的方案与代码进行对应，对代码进行梳理，搞清楚哪些部分完成了什么任务，下载数据集并对数据集进行描述
- [ ] 对作者设计的方案进行分析与验证：为什么要这么做？能否进行拓展？
- [ ] 按照要求总结成演示文稿+准备问答
- [ ] 撰写最终报告
