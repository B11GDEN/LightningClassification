def get_all_parent_layers(net, type):
    layers = []

    for name, l in net.named_modules():
        if isinstance(l, type):
            tokens = name.strip().split('.')

            layer = net
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]

            layers.append([layer, tokens[-1]])

    return layers


if __name__ == '__main__':
    '''Simple example how change attention type'''
    import torch
    from attention import LinearAttention
    from timm.models.vision_transformer import VisionTransformer, Attention
    from timm.models.deit import VisionTransformerDistilled

    deit = VisionTransformerDistilled()
    for parent_layer, last_token in get_all_parent_layers(deit, Attention):
        setattr(
            parent_layer, last_token,
            LinearAttention(
                dim=768, num_heads=12,
                qkv_bias=True, kv_drop=0., proj_drop=0.,
                q_kernel='l2', k_kernel='l2',
            )
        )
    x = torch.rand(16, 3, 224, 224)
    y = deit(x)
    print(y)
