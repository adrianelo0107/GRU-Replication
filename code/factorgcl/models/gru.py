"""FactorGCL附录金融GRU基线；未披露的层数/宽度明确作为补全。

采用PyTorch标准reset-after单元，无BN、图、因子或未来分支。
Cho正文/编码器reset-before参考式仅用于独立数值审计，不用于本次训练。
"""
from factorgcl import torch
from torch import nn


class FinancialGRU(nn.Module):
    cl_enabled = False

    def __init__(self, config, horizons):
        super().__init__()
        self.gru = nn.GRU(config.input_dim, config.embedding_dim,
                          num_layers=config.gru_layers, batch_first=True,
                          bias=True, dropout=0., bidirectional=False)
        self.head = nn.Linear(config.embedding_dim, horizons, bias=True)

    def forward_history(self, x, beta=None):
        # 每个股票窗口独立，未传hidden时每次从零状态开始。
        _, hidden = self.gru(x)
        return {'prediction': self.head(hidden[-1])}

    def predict(self, x, beta=None):
        return self.forward_history(x, beta)['prediction']

    def forward(self, x, beta=None):
        return self.predict(x, beta)


def make_model(config, horizons):
    assert config.name == 'gru'
    return FinancialGRU(config, horizons)


def reference_cell(x, h, weight_ih, weight_hh, bias_ih, bias_hh, reset_before=False):
    """独立展开r/z/n三门。before版本用于检验矩阵与逐元素乘不可交换。

    两种bias拆分沿用PyTorch，仅无bias时用于严格对应Cho省略bias的公式。
    """
    input_terms = x @ weight_ih.T + bias_ih
    hidden_terms = h @ weight_hh.T + bias_hh
    ir, iz, inn = input_terms.chunk(3, -1)
    hr, hz, hn = hidden_terms.chunk(3, -1)
    r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
    if reset_before:
        hidden_dim = h.shape[-1]
        hn = (r * h) @ weight_hh[2 * hidden_dim:].T + bias_hh[2 * hidden_dim:]
    else:
        hn = r * hn
    n = torch.tanh(inn + hn)
    return z * h + (1. - z) * n
