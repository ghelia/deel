import chainer
import chainer.functions as F
import chainer.links as L

"""
 Based on chainer official example 
 https://github.com/pfnet/chainer/tree/master/examples/ptb

 Modified by shi3z March 28,2016
"""
class RNNLM(chainer.Chain):

    """Recurrent neural net languabe model for penn tree bank corpus.

    This is an example of deep LSTM network for infinite length input.

    """
    def __init__(self, n_input_units=1000,n_vocab=100, n_units=100, train=True):
        super(RNNLM, self).__init__(
            inputVector= L.Linear(n_input_units, n_units),
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()

    def __call__(self, x,mode=0):
        if mode == 1:
            h0 = self.inputVector(x)
        else:
            h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
        return y

