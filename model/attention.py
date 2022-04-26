import torch
import torch.nn as nn

def buildListFromConfigs(config, inp_size, out_size, batch_norm=True, dropout=0.5, initialization=False, nonlinear=nn.SELU):
    layer_lst = []
    for i, v in enumerate(list(config)+[out_size]):
        if batch_norm and i > 0: #generally no batch norm before the model starts
            layer_lst.append(nn.LayerNorm(inp_size))
        if dropout > 1e-8:
            layer_lst.append(nn.Dropout(dropout))
        layer_lst.append(nn.Linear(inp_size, v, bias=True))
        if initialization:
            nn.init.kaiming_normal_(layer_lst[-1].weight,mode='fan_in',nonlinearity='linear')
        if i < len(config):
            layer_lst.append(nonlinear())
        inp_size = v
    return layer_lst, inp_size

class BuildLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, proj_size):
        self.num_layers = num_layers
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)]) #nn.ModuleList([LockedDropout(dropout) for i in range(num_layers)])
        layers = []
        for i in range(num_layers):
            layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        self.layers = nn.ModuleList(layers)
        self.final = nn.Linear(input_size, proj_size) if input_size != proj_size else nn.Identity
    
    def forward(self, data, hidden=None):
        if hidden is None:
            hidden = [None for i in range(self.num_layers)]
        for i in range(num_layers):
            data = self.dropout[i](data)
            data, hidden[i] = self.layers[i](data, hidden)
        data = self.final(data)
        return data, hidden
        

class ConditionalAutoEncoder(nn.Module):
    # starting macro is the state of the last 13 months. For training, set it to None (equivalent to setting it to a torch of zeros).
    # However, since testing data is low, run the macro net on the past <attention_span> months and send that. First month's data is not used.
    # Requires a <attention_span>*<n_macro_size> torch vector in the same device as the model, where n_macro_size is the output of the n_hidden_macro net.
    def __init__(self, n_alphas, n_factors, n_macro, n_hidden_macro, n_hidden_lhs, n_hidden_rhs, attention_vs_lstm='attention', starting_macro = None, starting_rhs = None,
                    batch_norm_lhs=True, batch_norm_rhs=False, dropout_p=0, initialization=True, rhs_input_assets = False, rhs_activation = True,
                    bn_momentum=0.1, bn_track_running_stats = True, macro_attention_span=1, rhs_attention_span=1, num_lstm_layers=1):
        # n_assets can be assets or portfolios:
            # In both cases, we are interested in asset return or portfolio return on the RHS
            # LHS is the original factor 
        super(ConditionalAutoEncoder, self).__init__()

        self.n_alphas = n_alphas
        self.n_factors = n_factors
        self.n_hidden_lhs = list(n_hidden_lhs)
        self.n_hidden_rhs = list(n_hidden_rhs)
        self.n_macro = n_macro
        self.n_hidden_macro = n_hidden_macro
        self.lstm = attention_vs_lstm != 'attention'

        if attention_vs_lstm == 'attention':
            self.macro_attention_span = macro_attention_span
            self.rhs_attention_span = rhs_attention_span
            attention_lst = buildListFromConfigs(n_hidden_macro, n_macro+macro_attention_span*n_factors + rhs_attention_span*n_factors, n_factors, batch_norm=True, dropout=dropout_p, initialization=True, nonlinear=nn.SELU)
            self.macro = nn.Sequential(attention_lst)
        else:
            attention_lst = buildListFromConfigs(n_hidden_macro, n_macro+macro_attention_span*n_factors + rhs_attention_span*n_factors, n_factors, batch_norm=True, dropout=dropout_p, initialization=True, nonlinear=nn.SELU)
            self.macro = nn.Sequential(attention_lst)
            self.macro_lstm = BuildLSTM(n_macro+macro_attention_span*n_factors + rhs_attention_span*n_factors, n_hidden_macro, num_lstm_layers, dropout=dropout_p, proj_size=n_macro)


        self.prev_macro = torch.zeros((macro_attention_span, n_factors), requires_grad=True) if starting_macro is None else starting_macro
        self.prev_rhs   = torch.zeros((rhs_attention_span, n_factors), requires_grad=True) if starting_rhs is None else starting_rhs

        
        lhs_lst = buildListFromConfigs(n_hidden_lhs, n_alphas, n_factors, batch_norm_lhs, dropout=dropout_p, initialization=True, nonlinear=nn.SELU)
        rhs_lst = buildListFromConfigs(n_hidden_rhs, n_alphas+1, n_factors, batch_norm_rhs, dropout=dropout_p, initialization=True, nonlinear=nn.SELU if rhs_activation else nn.Identity)
        if rhs_activation:
            rhs_lst.append(nn.SELU()) #Still not sold on this

        
        self.lhs = nn.Sequential(lhs_lst)
        self.rhs = nn.Sequential(rhs_lst)
    
    def run_through_lhs(self, data):
        #y_lhs_lst = []
        #for i in range(x_lhs.shape[2]):
        #    y_cur = self.lhs(x_lhs[:,:,i])
        #    y_cur = torch.unsqueeze(y_cur,2)
        #    y_lhs_lst.append(y_cur)
        #y_lhs = torch.cat(y_lhs_lst,dim=2)

        shape = data.shape
        data = data.permute(0, 2, 1).reshape(-1, shape[1])
        out = self.lhs(data)
        return out.reshape(shape[0], -1, shape[1]).permute(0, 2, 1)
    
    def run_through_macro(self, x_macro, prev_hidden):
        if self.lstm:
            y_macro = self.macro(x_macro)
            y_macro, hidden = self.macro_lstm(y_macro, prev_hidden)
        else:
            macro = torch.cat((x_macro, self.prev_macro.reshape(1, -1), self.prev_rhs.reshape(1, -1)))
            y_macro = self.macro(macro)
            hidden = None
        return y_macro, hidden

    def forward(self, x_macro, x_lhs, x_rhs, prev_hidden=None):
        y_lhs = self.run_through_lhs(x_lhs)
        y_rhs = self.rhs(x_rhs).unsqueeze(dim=2)
        self.last_rhs_factors = y_rhs
        y_macro, hidden = self.run_through_macro(y_macro, prev_hidden)
        out = torch.sum(y_lhs*y_rhs*y_macro, axis=2) #Double check this
        self.prev_macro = torch.roll(self.prev_macro, -1, 0)
        self.prev_macro[-1] = y_macro
        self.prev_rhs = torch.roll(self.prev_rhs, -1, 0)
        self.prev_rhs[-1] = y_rhs
        return out, y_rhs, hidden

    def forward_given_factors(self, x_macro, x_lhs, factors, prev_hidden=None):
        y_lhs = self.run_through_lhs(x_lhs)
        y_rhs = factors
        y_macro, hidden = self.run_through_macro(y_macro, prev_hidden)
        #out = torch.bmm(y_lhs.transpose(1,2), y_rhs).squeeze(2)
        out = torch.sum(y_lhs*y_rhs*y_macro, axis=2) #Double check this
        return out