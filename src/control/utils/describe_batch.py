import torch


def describe_batch(batch):
    if not isinstance(batch, tuple):
        return f"Not a tuple, got {type(batch)}"

    description = []
    for i, item in enumerate(batch):
        try:
            item_type = type(item).__name__
            if isinstance(item, list):
                inner_types = {type(x).__name__ for x in item}
                description.append(f"Item {i}: list[{', '.join(inner_types)}] (len={len(item)})")
            elif isinstance(item, torch.Tensor):
                description.append(f"Item {i}: Tensor {tuple(item.shape)}")
            else:
                description.append(f"Item {i}: {item_type}")
        except Exception as e:
            description.append(f"Item {i}: error during inspection: {e}")
    return "\n".join(description)