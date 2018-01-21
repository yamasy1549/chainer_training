##########
# (0) import
##########
import numpy as np
import chainer
from chainer import cuda, Function, \
        report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from sklearn import datasets

##########
# (1) データの準備
##########
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target.astype(np.int32)
N = Y.size
Y2 = np.zeros(3*N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i, Y[i]] = 1.0

index = np.arange(N)
xtrain = X[index[index % 2 != 0], :]
ytrain = Y2[index[index % 2 != 0]]
xtest = X[index[index % 2 == 0], :]
yans = Y[index[index % 2 == 0]]

##########
# (2) モデルの記述
##########
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1 = L.Linear(4,6),
            l2 = L.Linear(6,3),
        )
    def __call__(self,x,y):
        return F.softmax_cross_entropy(self.fwd(x),y)
    def fwd(self,x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2

##########
# (3) モデルと最適化アルゴリズムの設定
##########
model = IrisChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

##########
# (4) 学習
##########
n = 75
bs = 25
for j in range(2000):
    accum_loss = None
    sffindx = np.random.permutation(n)
    model.cleargrads()
    for i in range(0,n,bs):
        x = Variable(xtrain[i:(i+bs) if (i+bs) < n else n])
        y = Variable(ytrain[i:(i+bs) if (i+bs) < n else n])
        loss = model(x,y)
        loss.backward()
        accum_loss = loss if accum_loss is None else accum_loss + loss
    accum_loss.backward()
    optimizer.update()

##########
# (5) 評価
##########
xt = Variable(xtest)
yt = model.fwd(xt)
ans = yt.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    if cls == yans[i]:
        ok += 1

print(ok, "/", nrow, " = ", (ok * 1.0)/nrow)
