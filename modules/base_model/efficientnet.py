import torch 
import numpy as np 
import time
import os 
from efficientnet_pytorch import EfficientNet


class EfficientNet_ASPP(EfficientNet):
    # def __init__(self):
    #     super(EfficientNet_ASPP, self).__init__()
    
    def extract_features(self, inputs):
        """use convolution layer to extract feature .
        Note:
            If model is "efficientnet-b3"
            Get model backbone with multi layer output: 
            Mb block 13 (1 * 96 * 32 * 32)
            Mb block 17 ( 136 * 32 * 32)
            Mb block 21 ( 232  * 16 * 16)
            Mb block 24 (384 * 16  * 16)
            13 - 18 - 24 - 26
            
        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        l_block = []
        # Blocks
        l_extract_layer = [12, 17, 23, 25]
        for idx, block in enumerate(self._blocks):
            
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in l_extract_layer:
                l_block.append(x)

        return l_block
    
    
    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        # print("Extract feature")
        x = self.extract_features(inputs)
        
        # Pooling and final linear layer
        # x = self._avg_pooling(x)
        # if self._global_params.include_top:
        #     x = x.flatten(start_dim=1)
        #     x = self._fc(x)
        return x
    
    
if __name__ == '__main__':
    model_name = 'efficientnet-b3'
    device = torch.device("cuda")
    data = np.ones((1,3,512,512), dtype=np.float)
    input_tensor = torch.tensor(data, dtype= torch.float32)
    # input_tensor.to(device)
    model = EfficientNet_ASPP.from_pretrained(model_name)
    output = model(input_tensor)
    for block in output:
        print(block.shape)
    