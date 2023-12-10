import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, gen_nodes, out_dim, act=None, n_layers=None):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        # Num of nodes in each layer
        if isinstance(gen_nodes, list):
            n_nodes = gen_nodes
        elif isinstance(gen_nodes, int):
            n_nodes = [gen_nodes] * (n_layers - 1) + [out_dim]
        
        # Activation function for each layer
        if isinstance(act, list):
            act_func = act
        elif isinstance(act, str):
            act_func = [act] * (n_layers - 1) + ['linear']
        
        layers_list = []
        for ii in range(n_layers):
            if isinstance(act_func[ii], str):
                if act_func[ii] == 'lrelu':
                    layers_list.append(nn.LeakyReLU(0.1))
                else:
                    if act_func[ii].lower() == 'tanh':
                        activation_func = nn.Tanh()
                    elif act_func[ii].lower() == 'relu':
                        activation_func = nn.ReLU()
                    elif act_func[ii].lower() == 'linear':
                        activation_func = nn.Identity()
                    else:
                        raise NotImplementedError('Activation function {} is not implemented.'.format(act_func[ii]))
            else: 
                activation_func = act_func[ii]
                
            layers_list.extend([nn.Linear(input_dim if ii == 0 else n_nodes[ii - 1], n_nodes[ii]), activation_func])
        
        # self.layers = layers_list
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)