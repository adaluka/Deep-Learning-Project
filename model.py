import torch
import torch.nn as nn
from buildFromConfig import buildListFromConfigs

def buildListFromConfigs(config, inp_size, out_size, batch_norm=True, dropout=0.5, initialization=False, nonlinear=nn.SELU):
    layer_lst = []
    for i, v in enumerate(list(config)+[out_size]):
        if batch_norm and i > 0: #generally no batch norm before the model starts
            layer_lst.append(nn.LayerNorm(inp_size))
        if dropout > 1e-8 and i > 0: #generally no dropout before the model starts
            layer_lst.append(nn.Dropout(dropout))
        layer_lst.append(nn.Linear(inp_size, v, bias=True))
        if initialization:
            nn.init.kaiming_normal_(layer_lst[-1].weight,mode='fan_in',nonlinearity='linear')
        if i < len(config):
            layer_lst.append(nonlinear())
        inp_size = v
    return layer_lst

class Left(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, use_batch_norm=True, dropout=0.0, initialization=True, nonlinear = nn.SELU) -> None:
        super().__init__()
        
        lhs_lst = buildListFromConfigs(n_hidden, n_input, n_output, use_batch_norm, dropout, initialization, nonlinear)
        self.model = nn.Sequential(*lhs_lst)
    
    def forward(self, data):
        shape = data.shape
        data = data.permute(0, 2, 1).reshape(-1, shape[1])
        out = self.model(data)
        return out.reshape(shape[0], shape[2], -1).permute(0, 2, 1)

class Right(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, use_batch_norm=False, dropout=0.0, initialization=True, nonlinear = nn.SELU) -> None:
        super().__init__()
        
        rhs_lst = buildListFromConfigs(n_hidden, n_input, n_output, use_batch_norm, dropout, initialization, nonlinear)
        self.model = nn.Sequential(*rhs_lst)
    
    def forward(self, data):
        return self.model(data)

class RegimeMacro(nn.Module):
    def __init(self, n_input, n_output, n_layers, n_hidden_before=None, n_hidden_after=None, batch_norm=True, dropout=0.0, initialization=True, nonlinear=nn.SELU) -> None:
        super().__init__()
        if n_hidden_before is not None:
            self.model_before = nn.Sequential(*buildListFromConfigs(n_hidden_before, n_input, n_output, batch_norm, dropout, initialization, nonlinear))
            n_input = n_output
        else:
            self.model_before = nn.Identity()
        self.LSTM = nn.LSTM(n_input, n_output, n_layers, dropout=dropout)
        if n_hidden_after is not None:
            self.model_after = nn.Sequential(*buildListFromConfigs(n_hidden_after, n_input, n_output, batch_norm, dropout, initialization, nonlinear))
        else:
            self.model_after = nn.Identity()
    
    def forward(self, data):
        y_macro = self.model_before(data)
        y_macro, _ = self.LSTM(y_macro)
        y_macro = self.model_after(data)
        return y_macro


class model(nn.Module):
    def __init__(self, n_alphas, n_factors, n_macro, n_hidden_lhs, n_hidden_rhs, batch_norm_lhs=True, batch_norm_rhs=False, dropout_p=0, initialization=True, rhs_activation = True,
                    num_lstm_layers=1, nonlinear=nn.SELU, n_hidden_macro_before=None, n_hidden_macro_after=None, batch_norm_macro=True) -> None:
        super().__init__()
        self.left = Left(n_alphas, n_factors, n_hidden_lhs, batch_norm_lhs, dropout_p, initialization, nonlinear)
        self.right = Right(n_alphas+1, n_factors, n_hidden_rhs, batch_norm_rhs, dropout_p, initialization, nonlinear if rhs_activation else nn.Identity)
        self.lstm = RegimeMacro(n_macro, n_factors, num_lstm_layers, n_hidden_macro_before, n_hidden_macro_after, batch_norm_macro, dropout_p, initialization, nonlinear)
    
    def forward(self, x_macro, macro_idx, x_lhs, x_rhs, rhs=None):
        y_lhs = self.left(x_lhs)
        if rhs is None:
            rhs = self.right(x_rhs).unsqueeze(dim=2)
        y_macro = self.lstm(x_macro)[0, macro_idx, :][:, :, None]
        out = y_lhs*rhs*y_macro
        out = torch.sum(out, dim=1) #Double check this
        return out, rhs