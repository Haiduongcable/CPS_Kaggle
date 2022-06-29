import torch 
import torch.nn as nn 
import torch.functional as F 


class EncoderDecoderSegformer(nn.Module):
    def __init__(self, backbone, decode_head, neck = None, pretrained = None, representation = None):
        '''
        Encoder Decoder based Segformer
        Args: backbone: Module backbone MixVisionTransformer 
              decode_head: Module stack MLP layers 
              representation: represent feature to feature per pixel (a pixel be represented like a vector 256)
              pretrained: pretrained in ImageNet1k (MIT pretrained) or pretrained in ADE20K (ADE pretrained)
        
        '''
        super(EncoderDecoderSegformer, self).__init__()
        self.backbone = backbone 
        self.decode_head = decode_head
        if self.neck is not None: 
            self.neck = neck 
        self.pretrained = pretrained
        self.representation = representation
        
    def _init_backbone(self):
        pass 
    
    def __init_decode(self):
        pass 
    
    def __init_weight(self):
        if self.pretrained:
            #load pretrained 
            pass 
        pass
    
    def forward(self, input):
        '''
        Args: inputs: (b, c, h, w) torch float 32 (batch input of model)
        if train:
            return pred and representation for contrastive memory bank
        else inference, eval: 
            return pred
        '''
        _,_,H,W = input.shape
        feature_extract = self.backbone(input)
        if not self.training: 
            logits = self.decode_head(feature_extract)
            pred = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=True)
            return pred
        
        if self.representation is not None: 
            representation_vector = self.representation(feature_extract)
        logits = self.decode_head(feature_extract)
        pred = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
        
        return pred, representation_vector