import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import torch.optim as optimizer
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor

# get_data

sentences = ["jack like dog", "jack like cat", "jack like animal",
             "dog cat animal", "banana apple cat dog like", "dog fish milk like",
             "dog cat animal like", "jack like apple", "apple like", "jack like banana",
             "apple banana jack movie book music like", "cat dog hate", "cat dog like"]
s_l = " ".join(sentences).split()  # ['jack', 'like', 'dog']
vocab = list(set(s_l))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# parameter
C = 2  # window size
batch_size = 8
m = 2  # word embedding dim 向量维度

# print(word2idx[s_l[1]])

skip_grams = []
for idx in range(C, len(s_l) - C):
    center = word2idx[s_l[idx]]
    # print(center)
    # context = [word2idx[s_l[idx-2]],word2idx[s_l[idx-1]],word2idx[s_l[idx+1]],word2idx[s_l[idx+2]]]
    # print(context)
    context = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
    context = [word2idx[s_l[idx]] for idx in context]

    for w in context:
        skip_grams.append([center, w])


# print(skip_grams)

def make_data(skip_grams):
    input_data, output_data = [], []
    for a, b in skip_grams:
        input_data.append(
            np.eye(vocab_size)[a])  # 制作一个矩阵，输入one-hot #[array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
        output_data.append(b)  # 直接添加,输出是类别
    return input_data, output_data


input_data, output_data = make_data(skip_grams)
input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset, batch_size, True)


class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Parameter(torch.randn(vocab_size, m).type(dtype))
        self.V = nn.Parameter(torch.randn(m, vocab_size).type(dtype))

    def forward(self, X):
        # X:[batch_size ,vocab_size ]
        hidden = torch.mm(X, self.W)  # [batch_size,m]
        output = torch.mm(hidden, self.V)  # [batch_size,vocab_size]
        return output


model = Word2Vec().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optim = optimizer.Adam(model.parameters(), lr=1e-3)

for epoch in range(100000):
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        if (epoch + 1) % 1000 == 0:
            print(epoch + 1, i, loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()


def simlarityCalu(vector1, vector2):
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        simlarity = 0
    return simlarity


def word_vec(vocab):
    key_dict = {}
    for i, label in enumerate(vocab):
        W, WT = model.parameters()
        key_dict[label] = W[i].cpu().detach().numpy()
    return key_dict


def sen_vec(sentences):
    f_l = []
    for sen in sentences:
        f = 0
        for word in sen.split():
            f = f + dic[word]
        f_l.append(f)
    return f_l


dic = word_vec(vocab)
senvec = sen_vec(sentences)

sim_threshold = 0.9
# def sim_compare():
for n in range(len(senvec)):
    print(simlarityCalu(senvec[0], senvec[n]))



for i, label in enumerate(vocab):
    W, WT = model.parameters()
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
