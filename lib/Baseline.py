import torch

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
    
        r_out_1, (h_n, h_c) = self.lstm(x, None)
        return torch.transpose(h_n, 0, 1), torch.transpose(h_c, 0, 1)
    
    
class ConditionalGenerator(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, noise_size, translation=False):
        super(ConditionalGenerator, self).__init__()
        
        self.noise_size = noise_size
        self._translation = translation
        
        self.encoder = Encoder(input_size, hidden_size, num_layers)  
        self.cgenerator = torch.nn.LSTM(
            input_size=noise_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, steps):
        y0 = x[:, -1:, :]
        h, c = self.encoder(x)
        h0, c0 = torch.transpose(h, 0, 1), torch.transpose(c, 0, 1)
        
        
        if self._translation:
            z = torch.randn([x.shape[0], steps+1, self.noise_size]).to(x.device)
            gsamples, (h_n, h_c) = self.cgenerator(z, (h0.contiguous(), c0.contiguous()))
            y_pre = self.out(gsamples)
            return y_pre - y_pre[:, :1, :] + y0
        else:
            z = torch.randn([x.shape[0], steps, self.noise_size]).to(x.device)
            gsamples, (h_n, h_c) = self.cgenerator(z, (h0.contiguous(), c0.contiguous()))
            y_pre = self.out(gsamples)
            return torch.cat([y0, y_pre], dim=1)
 
    
class Critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sep):
        super(Critic, self).__init__()
        self._sep = sep
        self.lstm1 = torch.nn.LSTM(
            input_size=input_size[0],
            hidden_size=hidden_size[0],
            num_layers=num_layers[0],
            batch_first=True,
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=input_size[1],
            hidden_size=hidden_size[1],
            num_layers=num_layers[1],
            batch_first=True,
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size[0]+hidden_size[1], 1)
            )

    def forward(self, data):
        x, y = data[:, :self._sep, :], data[:, :, :]
        r_out_1, (h_n, h_c) = self.lstm1(x, None)
        r_out_2, (h_n, h_c) = self.lstm2(y, None)
        out = self.out(torch.cat([r_out_1[:, -1, :], r_out_2[:, -1, :]], dim=1))
        return out
    
