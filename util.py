import numpy as np 
from rl.core import Processor 
from rl.util import WhiteningNormalizer 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 


class Normalizerprocessor(Processor):
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalizer = None
    
    def state_batch_process(self, batch):
        batch_len = batch.shape[0]
        kernel = []
        for i in range(batch_len):
            observe = batch[i][..., :-ADDITIONAL_STATE]
            observe = self.scaler.fit_transform(observe)
            agent_state = batch[i][..., ADDITIONAL_STATE:]
            temp = np.concatenate((observe, agent_state), axis=1)
            temp = temp.reshape((1,) + temp.shape)
            kernel.append(temp)
        batch = np.concatenate(tuple(kernel))
        return batch