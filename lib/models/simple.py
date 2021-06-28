'''

'''

from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)
        fused_h, txt_h  = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask) 
        prediction = self.pred_layer(fused_h) * map_mask
        '''
        print('\n------- MODEL FORWARD with feature extraction')
        print('TENSORS SHAPES')
        print(f'visual_input.shape {visual_input.shape}')
        print(f'visual_input.transpose(1, 2).shape {visual_input.transpose(1, 2).shape}')
        print(f'vis_h.shape {vis_h.shape}')
        print(f'map_mask.shape {map_mask.shape}')
        print(f'map_h.shape {map_h.shape}')
        print(f'textual_input.shape {textual_input.shape}')   
        print(f'textual_mask.shape {textual_mask.shape}')  
        print(f'fused_h.shape {fused_h.shape}')  
        print(f'prediction.shape {prediction.shape}')
        '''
        '''
        print('TENSORS VALUES')
        print(f'visual_input: {visual_input}')
        print(f'visual_input.transpose(1, 2): {visual_input.transpose(1, 2)}')
        print(f'vis_h: {vis_h}')
        print(f'map_mask: {map_mask}')
        print(f'map_h: {map_h}')
        print(f'textual_input: {textual_input}')
        print(f'textual_mask: {textual_mask}')
        print(f'fused_h: {fused_h}')
        print(f'prediction: {prediction}')
        '''
        return fused_h, prediction, map_mask