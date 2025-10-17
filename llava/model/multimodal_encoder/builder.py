import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .adaptive_clip_encoder import AdaptiveCLIPVisionTower
from .multi_layer_clip_encoder import MultiLayerCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    # 新增逻辑：检查是否使用多层特征提取
    use_multi_layer = getattr(vision_tower_cfg, 'mm_vision_layers_to_use', None) is not None

    if use_multi_layer:
        return MultiLayerCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if "mPLUG" in vision_tower:
        return MPLUGVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    use_adaptive_layer_selection = getattr(vision_tower_cfg, 'use_adaptive_layer_selection', False)
    if use_adaptive_layer_selection:
        return AdaptiveCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
