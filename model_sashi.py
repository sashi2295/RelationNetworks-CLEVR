import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class ConvInputModel(nn.Module):

    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)

        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)

        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

    def forward(self, img):

        x = self.conv1(img)
        x = self.batchNorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)

        return x


class QuestionEmbedModel(nn.Module):

    def __init__(self, in_size, embed=32, hidden=128):

        super(QuestionEmbedModel, self).__init__()

        self.wembedding = nn.Embedding(in_size + 1, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.hidden = hidden

    def forward(self, question):

        wembed = self.wembedding(question)

        self.lstm.flatten_parameters()
        _, hidden = self.lstm(wembed)

        qst_emb = hidden[0]
        qst_emb = qst_emb[0]

        return qst_emb

class RelationalLayerBase(nn.Module):

    def __init__(self, in_size, out_size, qst_size, hyp):
        super().__init__()

        self.f_fc1 = nn.Linear(hyp["g_layers"][-1], hyp["f_fc1"])
        self.f_fc2 = nn.Linear(hyp["f_fc1"], hyp["f_fc2"])
        self.f_fc3 = nn.Linear(hyp["f_fc2"], out_size)

        self.dropout = nn.Dropout(p=hyp["dropout"])

        self.on_gpu = False
        self.hyp = hyp
        self.qst_size = qst_size
        self.in_size = in_size
        self.out_size = out_size

    def cuda(self):
        self.on_gpu = True
        super().cuda()


class RelationalLayer(RelationalLayerBase):
    def __init__(self, in_size, out_size, qst_size, hyp, extraction=False):
        super().__init__(in_size, out_size, qst_size, hyp)

        self.quest_inject_position = hyp["question_injection_position"]
        self.in_size = in_size

        self.g_layers = []
        self.g_layers_size = hyp["g_layers"]

        for idx, g_layer_size in enumerate(hyp["g_layers"]):
            in_s = in_size if idx==0 else hyp["g_layers"][idx-1]
            out_s = g_layer_size
            if idx==self.quest_inject_position:
                l = nn.Linear(in_s+qst_size, out_s)
            else:
                l = nn.Linear(in_s, out_s)

            self.g_layers.append(l)

        self.g_layers = nn.ModuleList(self.g_layers)
        self.extraction = extraction

    def forward(self, x, qst):

        b, d, k = x.size()
        qst_size = qst.size()[1]

        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1, d, 1)
        qst = torch.unsqueeze(qst, 2)

        x_i = torch.unsqueeze(x, 1)
        x_i = x_i.repeat(1, d, 1, 1)
        x_j = torch.unsqueeze(x, 2)
        x_j = x_j.repeat(1, 1, d, 1)

        x_full = torch.cat([x_i, x_j], 3)

        x_ = x_full.view(b * d **2, self.in_size)

        for idx, (g_layer, g_layer_size) in enumerate(zip(self.g_layers, self.g_layers_size)):
            if idx == self.quest_inject_position:
                in_size = self.in_size if idx==0 else self.g_layers_size[idx-1]

                x_img = x_.view(b, d, d, in_size)
                qst = qst.repeat(1, 1, d, 1)
                x_concat = torch.cat([x_img, qst], 3)

                x_ = x_concat.view(b*(d**2), in_size+self.qst_size)
                x_ = g_layer(x_)
                x_ = F.relu(x_)

            if self.extraction:
                return None

            x_g = x_.view(b, d**2, self.g_layers_size[-1])
            x_g = x_g.sum(1).squeeze(1)

            x_f = self.f_fc1(x_g)
            x_f = F.relu(x_f)
            x_f = self.f_fc2(x_f)
            x_f = self.dropout(x_f)
            x_f = F.relu(x_f)
            x_f = self.f_fc3(x_f)

            return F.log_softmax(x_f, dim=1)


class RN(nn.Module):

    def __init__(self, args, hyp, extraction=False):
        super(RN, self).__init__()
        self.coord_tensor = None
        self.on_gpu = False

        self.conv = ConvInputModel()
        self.state_desc = hyp['state_description']

        hidden_size = hyp["lstm_hidden"]
        self.text = QuestionEmbedModel(args.qdict_size, embed=hyp["lstm_word_emb"], hidden=hidden_size)

        self.rl_in_size = hyp["rl_in_size"]
        self.rl_out_size = args.adict_size
        self.rl = RelationalLayer(self.rl_in_size, self.rl_out_size, hidden_size, hyp, extraction)

        if hyp["question_injection_position"] != 0:
            print('Supposing IR model')
        else:
            print('Supposing original DeepMind model')

    def forward(self, img, qst_idxs):
        if self.state_desc:
            x = img

        else:
            x = self.conv(img)
            b, k, d, _ = x.size()
            x = x.view(b, k, d*d)

            if self.coord_tensor is None or torch.cuda.device_count() == 1:
                self.build_coord_tensor(b, d)
                self.coord_tensor = self.coord_tensor.view(b, 2, d*d)

            x = torch.cat([x, self.coord_tensor], 1)
            x = x.permute(0, 2, 1)

        qst = self.text(qst_idxs)
        y = self.rl(x, qst)
        return y

    def build_coord_tensor(self, b, d):
        coords = torch.linspace(-d/2., d/2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)

        ct = torch.stack((x, y))

        ct = ct.unsqueeze(0).repeat(b, 1, 1, 1)
        self.coord_tensor = Variable(ct, requires_grad=False)

        if self.on_gpu:
            self.coord_tensor = self.coord_tensor.cuda()

    def cuda(self):
        self.on_gpu = True
        self.rl.cuda()
        super(RN, self).cuda()



