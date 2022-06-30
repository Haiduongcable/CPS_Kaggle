import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from modules.base_model.mix_visiontransformer import mit_b0, mit_b1, mit_b2
from modules.base_model.decoder_segformer_representation import SegFormerHead

class EncoderDecoderSegformer(nn.Module):
    def __init__(self, backbone_cfg, pretrained = None, feature_strides=[4, 8, 16, 32],\
                in_channels=[64, 128, 320, 512],dropout_ratio=0.1,\
                decoder_params=dict(embed_dim=768), num_classes=150,\
                in_index = [0, 1, 2, 3]):
        '''
        Encoder Decoder based Segformer
        Args: backbone: Module backbone MixVisionTransformer 
              decode_head: Module stack MLP layers 
              representation: represent feature to feature per pixel (a pixel be represented like a vector 256)
              pretrained: pretrained in ImageNet1k (MIT pretrained) or pretrained in ADE20K (ADE pretrained)
        
        '''
        super(EncoderDecoderSegformer, self).__init__()
        if backbone_cfg == 'mitb2':
            self.backbone = mit_b2()
        
        self.pretrained = pretrained
        
        self.backbone = self.load_checkpoint() 
        self.decode_head = SegFormerHead(feature_strides=[4, 8, 16, 32],\
                                in_channels=[64, 128, 320, 512],\
                                dropout_ratio=0.1,\
                                decoder_params=dict(embed_dim=768),\
                                num_classes=num_classes,\
                                in_index = [0, 1, 2, 3])

        
    def load_checkpoint(self):
        '''
        Args: model
        '''
        state_dict_checkpoint = torch.load(self.pretrained)
        
        self.backbone.load_state_dict(state_dict_checkpoint, strict=False)
        ckpt_keys = set(state_dict_checkpoint.keys())
        own_keys = set(self.backbone.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        if len(missing_keys) == 0:
            print('Load pretrained success')
        del state_dict_checkpoint

        return self.backbone
    
    def forward(self, input):
        '''
        Args: inputs: (b, c, h, w) torch float 32 (batch input of model)
        if train:
            return pred and representation for contrastive memory bank
        else inference, eval: 
            return pred
        '''
        _,_,H,W = input.shape
        l_feature_extract = self.backbone(input)
        if not self.training: 
            logits = self.decode_head(l_feature_extract)
            pred = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=True)
            return pred
        logits, rep = self.decode_head(l_feature_extract)
        pred = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=True)
        
        return pred, rep 
    
if __name__ == '__main__':
    device = torch.device("cuda")
    path_checkpoint = '/home/asilla/duongnh/project/Analys_COCO/tmp_folder/CrossPseudo_UpdateBranch/CPS_Kaggle/pretrained/mit_b2.pth'
    model = EncoderDecoderSegformer(backbone_cfg='mitb2', pretrained=path_checkpoint)
    model.to(device)
   
    # model.to(device)
    #backbone.eval()
    # summary(backbone, (3,128,128))
    input_image = torch.randn(4, 3, 512, 512)
    input_image = input_image.to(device)
    pred, rep = model(input_image)
    print(pred.shape)
    print(rep.shape)