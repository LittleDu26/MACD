# 蒸馏目标

设父代与子代的共享体素集合为

$$
\Omega_{p,c}
=
\{k\mid x_p(k)=x_c(k),\ x_p(k)\neq 0\}
$$

记共享体素数量为

$$
n_s = |\Omega_{p,c}|
$$

对于任意蒸馏样本 $o_t$，记父代和子代控制器在第 $l$ 层、第 $h$ 个注意力头得到的注意力矩阵分别为

$$
A_{p,t}^{l,h},\qquad A_{c,t}^{l,h}
$$

同时记第 $l$ 层 Transformer 输出的隐藏表示为

$$
H_{p,t}^{l},\qquad H_{c,t}^{l}
$$

蒸馏仅在共享体素集合 $\Omega_{p,c}$ 上进行，目标同时约束父子控制器在共享结构上的注意力交互模式与隐藏特征表示。

## 1. 共享注意力蒸馏

对于第 $l$ 层、第 $h$ 个注意力头，仅提取共享体素之间的注意力子矩阵：

$$
A_{u,t,\Omega}^{l,h}
=
A_{u,t}^{l,h}
[\Omega_{p,c},\Omega_{p,c}],
\qquad
u\in\{p,c\}
$$

其中

$$
A_{u,t,\Omega}^{l,h}
\in
\mathbb{R}^{n_s\times n_s}
$$

矩阵第 $i$ 行表示共享体素 $i$ 作为 query 时，对其他共享体素的注意力分布。

若完整注意力矩阵是在全部 token 上 softmax 后再截取共享区域，则截取后的每一行需要重新归一化：

$$
\hat{A}_{u,t}^{l,h}(i,j)
=
\frac{
A_{u,t}^{l,h}(i,j)
}{
\sum_{k\in\Omega_{p,c}}
A_{u,t}^{l,h}(i,k)
+\varepsilon
},
\qquad
i,j\in\Omega_{p,c}
$$

其中 $\varepsilon$ 为数值稳定项。

若前向传播时已通过 attention mask 限制共享体素仅与共享体素发生注意力交互，则可直接使用共享子矩阵中的行分布。

对于任意共享体素 $i\in\Omega_{p,c}$，父代对应的注意力分布作为 teacher，子代对应的注意力分布作为 student。逐 query 的注意力蒸馏项为

$$
D_{\mathrm{KL}}
\left(
\hat{A}_{p,t}^{l,h}(i,\Omega_{p,c})
\middle\|
\hat{A}_{c,t}^{l,h}(i,\Omega_{p,c})
\right)
$$

展开为

$$
\sum_{j\in\Omega_{p,c}}
\hat{A}_{p,t}^{l,h}(i,j)
\log
\frac{
\hat{A}_{p,t}^{l,h}(i,j)
}{
\hat{A}_{c,t}^{l,h}(i,j)+\varepsilon
}
$$

因此，对单个样本 $o_t$，共享注意力蒸馏损失定义为

$$
\mathcal{L}_{\mathrm{attn}}^{(t)}
=
\frac{1}{LHn_s}
\sum_{l=1}^{L}
\sum_{h=1}^{H}
\sum_{i\in\Omega_{p,c}}
D_{\mathrm{KL}}
\left(
\hat{A}_{p,t}^{l,h}(i,\Omega_{p,c})
\middle\|
\hat{A}_{c,t}^{l,h}(i,\Omega_{p,c})
\right)
$$

该损失直接约束每个共享体素的注意力分布，使父代和子代中同位置、同类型体素保持相似的共享结构信息交互模式，而不再将整个注意力矩阵压缩为单一的列重要性统计量。

对于 mini-batch $\mathcal{B}$，batch 级注意力蒸馏损失为

$$
\mathcal{L}_{\mathrm{attn}}
=
\frac{1}{|\mathcal{B}|}
\sum_{o_t\in\mathcal{B}}
\mathcal{L}_{\mathrm{attn}}^{(t)}
$$

## 2. 共享特征蒸馏

仅约束注意力分布一致，并不能保证父子控制器最终形成相同的 token 表示，因此进一步对共享体素在 Transformer 中的隐藏特征进行对齐。

记第 $l$ 层中位置 $i$ 的隐藏表示为

$$
H_{u,t,i}^{l}
\in
\mathbb{R}^{d},
\qquad
u\in\{p,c\}
$$

其中 $d$ 为隐藏特征维度。

对任意共享体素 $i\in\Omega_{p,c}$，要求子代隐藏表示接近父代对应位置的隐藏表示。单个共享体素的特征蒸馏项定义为

$$
\left\|
\bar{H}_{p,t,i}^{l}
-
\bar{H}_{c,t,i}^{l}
\right\|_2^2
$$

其中 $\bar{H}$ 表示用于蒸馏的归一化隐藏特征。若直接提取 Transformer 中 LayerNorm 后的表示，则可令

$$
\bar{H}_{u,t,i}^{l}
=
H_{u,t,i}^{l}
$$

否则可显式使用

$$
\bar{H}_{u,t,i}^{l}
=
\operatorname{LN}
\left(
H_{u,t,i}^{l}
\right)
$$

以减弱隐藏特征绝对尺度差异对蒸馏的影响。

对于单个样本 $o_t$，共享特征蒸馏损失定义为

$$
\mathcal{L}_{\mathrm{feat}}^{(t)}
=
\frac{1}{Ln_s}
\sum_{l=1}^{L}
\sum_{i\in\Omega_{p,c}}
\left\|
\bar{H}_{p,t,i}^{l}
-
\bar{H}_{c,t,i}^{l}
\right\|_2^2
$$

该损失约束父子控制器在共享体素上的内部表示，使同位置、同类型体素在经过多层 Transformer 信息聚合后仍保持相近的特征表达。

对于 mini-batch $\mathcal{B}$，batch 级共享特征蒸馏损失为

$$
\mathcal{L}_{\mathrm{feat}}
=
\frac{1}{|\mathcal{B}|}
\sum_{o_t\in\mathcal{B}}
\mathcal{L}_{\mathrm{feat}}^{(t)}
$$

## 3. 总蒸馏损失

共享注意力蒸馏与共享特征蒸馏分别约束共享体素的信息交互关系和隐藏表示结果。二者共同构成出生蒸馏目标：

$$
\mathcal{L}_{\mathrm{birth}}
=
\lambda_A
\mathcal{L}_{\mathrm{attn}}
+
\lambda_H
\mathcal{L}_{\mathrm{feat}}
$$

其中

$$
\lambda_A\geq 0,
\qquad
\lambda_H\geq 0
$$

分别表示共享注意力蒸馏和共享特征蒸馏的权重。

父代控制器仅作为固定 teacher 提供目标注意力分布与隐藏表示，因此父代输出在计算蒸馏损失时停止梯度传播：

$$
\hat{A}_{p}
=
\operatorname{stopgrad}
\left(
\hat{A}_{p}
\right)
$$

$$
\bar{H}_{p}
=
\operatorname{stopgrad}
\left(
\bar{H}_{p}
\right)
$$

因此，蒸馏损失仅对子代控制器参数产生梯度：

$$
\nabla_{\theta_p}
\mathcal{L}_{\mathrm{birth}}
=
0
$$

$$
\nabla_{\theta_c}
\mathcal{L}_{\mathrm{birth}}
\neq
0
$$

最终，$\mathcal{L}_{\mathrm{attn}}$ 用于保持共享体素之间的注意力组织方式，$\mathcal{L}_{\mathrm{feat}}$ 用于保持共享体素的内部特征表示，从而从关系层和表示层同时完成父代共享结构知识向子代的迁移。
