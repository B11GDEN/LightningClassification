import time
import torch
from src.model.components import LinearAttention, get_all_parent_layers
from timm.models.vision_transformer import VisionTransformer, Attention


def main():
    net = VisionTransformer(
        embed_dim=192,
        depth=12,
        num_heads=3,
    )
    for parent_layer, last_token in get_all_parent_layers(net, Attention):
        setattr(
            parent_layer, last_token,
            LinearAttention(
                dim=192,
                num_heads=64,
                q_kernel='l2',
                k_kernel='l2',
            )
        )
    x = torch.rand(128, 3, 224, 224)
    start = time.time()
    with torch.no_grad():
        y = net(x)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()