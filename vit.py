
from mindspore import nn, ops
import mindspore as ms
from typing import Optional
from mindspore.common.initializer import Normal
from mindspore.common.initializer import initializer
from mindspore import Parameter

def init(init_type, shape, dtype, name, requires_grad):
    """Init."""
    initial = initializer(init_type, shape, dtype).init_data()
    return Parameter(initial, name=name, requires_grad=requires_grad)

class ViT(nn.Cell):
    def __init__(self,
                 image_size: int = 224,
                 input_channels: int = 3,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_dim: int = 3072,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: Optional[nn.Cell] = nn.LayerNorm,
                 pool: str = 'cls',
                 num_classes=1000) -> None:
        super(ViT, self).__init__()

        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              embed_dim=embed_dim,
                                              input_channels=input_channels)
        num_patches = self.patch_embedding.num_patches

        # 此处增加class_embedding和pos_embedding，如果不是进行分类任务
        # 可以只增加pos_embedding，通过pool参数进行控制
        self.cls_token = init(init_type=Normal(sigma=1.0),
                              shape=(1, 1, embed_dim),
                              dtype=ms.float32,
                              name='cls',
                              requires_grad=True)

        # pos_embedding也是一组可以学习的参数，会被加入到经过处理的patch矩阵中
        self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                  shape=(1, num_patches + 1, embed_dim),
                                  dtype=ms.float32,
                                  name='pos_embedding',
                                  requires_grad=True)

        # axis=1定义了会在向量的开头加入class_embedding
        self.concat = ops.Concat(axis=1)

        self.pool = pool
        self.pos_dropout = nn.Dropout(keep_prob)
        self.norm = norm((embed_dim,))
        self.tile = ops.Tile()
        self.transformer = TransformerEncoder(dim=embed_dim,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              mlp_dim=mlp_dim,
                                              keep_prob=keep_prob,
                                              attention_keep_prob=attention_keep_prob,
                                              drop_path_keep_prob=drop_path_keep_prob,
                                              activation=activation,
                                              norm=norm)
        self.dropout = nn.Dropout(keep_prob)
        self.dense = nn.Dense(embed_dim, num_classes)

    def construct(self, x):
        """ViT construct."""
        x = self.patch_embedding(x)

        # class_embedding主要借鉴了BERT模型的用于文本分类时的思想
        # 在每一个word vector之前增加一个类别值，通常是加在向量的第一位
        cls_tokens = self.tile(self.cls_token, (x.shape[0], 1, 1))
        x = self.concat((cls_tokens, x))
        x += self.pos_embedding

        x = self.pos_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)

        # 增加的class_embedding是一个可以学习的参数，经过网络的不断训练
        # 最终以输出向量的第一个维度的输出来决定最后的输出类别；
        x = x[:, 0]

        if self.training:
            x = self.dropout(x)
        x = self.dense(x)
        return x

class Attention(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = ms.Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(keep_prob)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """Attention construct."""
        b, n, c = x.shape

        # 最初的输入向量首先会经过Embedding层映射成Q(Query)，K(Key)，V(Value)三个向量
        # 由于是并行操作，所以代码中是映射成为dim*3的向量然后进行分割
        qkv = self.qkv(x)

        #多头注意力机制就是将原本self-Attention处理的向量分割为多个Head进行处理
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)

        # 自注意力机制的自注意主要体现在它的Q，K，V都来源于其自身
        # 也就是该过程是在提取输入的不同顺序的向量的联系与特征
        # 最终通过不同顺序向量之间的联系紧密性（Q与K乘积经过Softmax的结果）来表现出来
        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 其最终输出则是通过V这个映射后的向量与QK经过Softmax结果进行weight sum获得
        # 这个过程可以理解为在全局上进行自注意表示
        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.out(out)
        out = self.out_drop(out)
        

        return out

class FeedForward(nn.Cell):
    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Cell = nn.GELU,
                 keep_prob: float = 1.0):
        super(FeedForward, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Dense(in_features, hidden_features)
        self.activation = activation()
        self.dense2 = nn.Dense(hidden_features, out_features)
        self.dropout = nn.Dropout(keep_prob)

    def construct(self, x):
        """Feed Forward construct."""
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x


class ResidualCell(nn.Cell):
    def __init__(self, cell):
        super(ResidualCell, self).__init__()
        self.cell = cell

    def construct(self, x):
        """ResidualCell construct."""
        return self.cell(x) + x


class TransformerEncoder(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_layers: int,
                 num_heads: int,
                 mlp_dim: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):
        super(TransformerEncoder, self).__init__()
        layers = []

        # 从vit_architecture图可以发现，多个子encoder的堆叠就完成了模型编码器的构建
        # 在ViT模型中，依然沿用这个思路，通过配置超参数num_layers，就可以确定堆叠层数
        for _ in range(num_layers):
            normalization1 = norm((dim,))
            normalization2 = norm((dim,))
            attention = Attention(dim=dim,
                                  num_heads=num_heads,
                                  keep_prob=keep_prob,
                                  attention_keep_prob=attention_keep_prob)

            feedforward = FeedForward(in_features=dim,
                                      hidden_features=mlp_dim,
                                      activation=activation,
                                      keep_prob=keep_prob)

            # ViT模型中的基础结构与标准Transformer有所不同
            # 主要在于Normalization的位置是放在Self-Attention和Feed Forward之前
            # 其他结构如Residual Connection，Feed Forward，Normalization都如Transformer中所设计
            layers.append(
                nn.SequentialCell([
                    # Residual Connection，Normalization的结构可以保证模型有很强的扩展性
                    # 保证信息经过深层处理不会出现退化的现象，这是Residual Connection的作用
                    # Normalization和dropout的应用可以增强模型泛化能力
                    ResidualCell(nn.SequentialCell([normalization1,
                                                    attention])),

                    ResidualCell(nn.SequentialCell([normalization2,
                                                    feedforward]))
                ])
            )
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        """Transformer construct."""
        return self.layers(x)


class PatchEmbedding(nn.Cell):
    MIN_NUM_PATCHES = 4
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 input_channels: int = 3):
        super(PatchEmbedding, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 通过将输入图像在每个channel上划分为16*16个patch
        self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        """Path Embedding construct."""
        x = self.conv(x)
        b, c, h, w = x.shape

        # 再将每一个patch的矩阵拉伸成为一个1维向量，从而获得了近似词向量堆叠的效果；
        x = self.reshape(x, (b, c, h * w))
        x = self.transpose(x, (0, 2, 1))

        return x
