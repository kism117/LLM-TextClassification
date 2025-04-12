import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from transformers import AutoModelForSequenceClassification

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        print('Mish activation loaded')

    def forward(self,x):
        out = F.softplus(x)
        x = x*(torch.tanh(softplus(x)))
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, output_size,init_method='xavier_uniform',bias=False):
        super(FeedForwardNetwork, self).__init__()
        # 定义全连接层
        if not bias:
            self.fc = nn.Linear(input_size, output_size,bias=False)
        else:
            self.fc = nn.Linear(input_size, output_size)
    

        # 参数初始化
        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.fc.weight)
            if bias:
                self.fc.bias.data.fill_(0)  # 初始化偏置为0
        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
            if bias:
                self.fc.bias.data.fill_(0)  # 初始化偏置为0
        else:
            raise ValueError('Unsupported initialization method')

    def forward(self, x):
        out = self.fc(x)
        return out

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        

        # 定义门控和线性转换层
        self.input_gate = FeedForwardNetwork(input_size + hidden_size, hidden_size)
        self.update_gate = FeedForwardNetwork(input_size + hidden_size, hidden_size)
        self.reset_gate = FeedForwardNetwork(input_size + hidden_size, hidden_size)

        # 激活函数为sigmoid和tanh
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input_, hidden_state):
        combined = torch.cat((input_, hidden_state), dim=2)  # torch.Size([2, 512, 4096])

        # 计算更新门和重置门信号
        z = self.sigmoid(self.update_gate(combined))
        r = self.sigmoid(self.reset_gate(combined))
        # 计算候选隐藏状态
        h_prime = self.tanh(self.input_gate(torch.cat((input_, r * hidden_state), dim=2)))
        # 更新隐藏状态
        hidden_state = (1 - z) * hidden_state + z * h_prime
        return hidden_state


class TextCNN(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.mish = Mish()
        self.convs = nn.ModuleList([nn.Conv2d(3, 64, (k, in_features), padding = (k//2, 0)) for k in [3,5,7]])
        self.dropout = nn.Dropout(0.5)
        self.fc = FeedForwardNetwork(64*3,out_features)

    def conv_and_pool(self, x, conv):
        x = self.mish(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        #out = x.unsqueeze(1)
        out = x
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        #out = out.permute(0, 2, 1)
        #print("out.shape ",out.shape)
        out = self.dropout(out)
        out = self.fc(out)
        #out = torch.index_select(out, dim=1, index=indices) 
        
        return out

class TextCNN_linear(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.L1 = torch.nn.Linear(in_features, out_features, bias=False)
        #self.L1 = FeedForwardNetwork(in_features,out_features)

    def forward(self, x):
        print("x shape ",x.shape)
        out = self.L1(x)
        print("out shape ",out.shape)
        return out

def create_and_replace(model):
    if hasattr(model, 'score'):
        target_model = model.score
        in_features = target_model.in_features
        out_features = target_model.out_features
        print(in_features,out_features)
        new_model = TextCNN(in_features,out_features)
        #new_model = TextCNN_linear(in_features,out_features)
        print("set_attr")
        setattr(model,"score",new_model)
    else:
        raise Exception("Please confirm whether the name of the layer in the model is correct")



if __name__=="__main__":
    model_path = "ckpt/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"
    num_labels = 7
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=num_labels)
    create_and_replace(model)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter Name: {name}, param is {param.dtype}")
            s = input()
        else:
            print(f"Parameter Name: {name}, Updateable: False")
    
    
    
    