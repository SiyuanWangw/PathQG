import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, query_size, memory_size, attn_size, output_concat=False):
        super(BahdanauAttention, self).__init__()
        self.query_size = query_size
        self.memory_size = memory_size
        self.attn_size = attn_size

        self.linear_query = nn.Linear(in_features=self.query_size, out_features=self.attn_size, bias=False)
        self.linear_memory = nn.Linear(in_features=self.memory_size, out_features=self.attn_size, bias=False)
        self.v = nn.Linear(self.attn_size, 1, bias=False)

    def forward(self, query, memory):
        w_query = self.linear_query(query)
        w_memory = self.linear_memory(memory)
        # print('query and memory size', query.size(), memory.size())

        extendded_query = w_query.unsqueeze(1)
        # print('size of extendded_query', extendded_query.size())
        alignment = self.v(torch.tanh(extendded_query.expand_as(w_memory) + w_memory))
        # print('alignment size', alignment.size())

        alpha = F.softmax(alignment.squeeze(2), -1)
        context = torch.matmul(alpha.unsqueeze(1), memory)
        return alpha, context




