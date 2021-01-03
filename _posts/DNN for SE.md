# DNN FOR SE

### Singing Voice Extraction with Attention-based Spectrograms Fusion

这篇文章的三个重要的贡献点：

1. In order to obtain better continuity of spectrogram, we **add a regular term to loss function**.
2. In order to alleviate inconsistent spectrogram, we **use the phase of the linear fusion waveform to reconstruct the final waveform**, because the iterative signal reconstruction can produce better resynthesized speech.
3. In order to get better neural network modeling capabilities, **attention mechanism is adopted**. We have tried a variety of embedding methods of multi spectrograms as the input of attention mechanism.

看一下主干模型

![image-20201204154051027](D:\OneDrive\webProject\HieDean.github.io\_posts\image-20201204154051027.png)

主干模型分为两部分，在第一部分中，带噪的时频谱通过双向LSTM得到两个输出，一个是直接映射得到的时频谱，另一个是通过预测mask得到的时频谱

##### Attention机制

第二部分中，首先要使用attention机制对first stage得到的两种时频谱进行处理，attention机制的处理可以有多种形式，文中讨论以下几种形式

![image-20201204155209966](D:\OneDrive\webProject\HieDean.github.io\_posts\image-20201204155209966.png)

其中h代表hidden layer，attention map的计算方式如下

$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})$

上式相当于计算出了一系列weight，attention机制的最终结果应当是computing a weighted sum of values (V)

经过attention之后的两个输出仍然可以被看作两种时频谱，这两种时频谱将被用于计算MDM

##### Minimum difference masks

这是文章新提出的一种mask，它与第一个贡献点中那个新加的loss项有关

首先看看什么是MDM

$d_i(t,f)=|spc_i(t,f)-spc_c(t,f)|$

$$\widetilde{MDM_i}(t,f)\begin{equation} \left\{ \begin{array}{lr} 1, \ \ \ i=argmin_id_i(t,f) &\\ 0, \ \ \ otherwise &\end{array} \right. \end{equation}$$

![image-20201204161525001](D:\OneDrive\webProject\HieDean.github.io\_posts\image-20201204161525001.png)

通过以上，可以计算出两种MDM，接下来是两种selection (这里的f是一个frequency bin)

$select_i(t,f)=MDM_i(t,f)*spc_i(t,f)$

$spc_f=\sum_i{select_i}$

最后fuse：$spc_{LSF}=(spc_{mapping}+spc_{masking})/2$

##### 相位

关于再相位上的贡献点，文中说

> Finally, we use the nonlinear fused spectrogram and the phase from the linear fusion constructed waveform to reconstruct the final enhanced waveform.

其中的linear fusion constructed waveform似乎来源于另一篇文章：Multiple-target deep learning for lstm-rnn based speech enhancement

##### 损失函数

$L_{MTL}=L_{mapping}+\alpha L_{masking}$

$L_{MDM}=\sum_i\sum_{t,f}{(MDM_i(t,f)-\widetilde{MDM}_i(t,f))^2}+\beta (L_{masking}+L_{mapping})$

$L_{MDM-tend}=L_{MDM}+\gamma (spc_f-spc_c)^2$



