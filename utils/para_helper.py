import os, torch

def load_state(path, model, optimizer=None, key="state_dict"):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    print(
                        "caution: size-mismatch key: {} size: {} -> {}".format(
                            k, v.shape, v_dst.shape
                        )
                    )

        for k in ignore_keys:
            state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)

        ckpt_keys = set(state_dict.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            best_metric = checkpoint["best_miou"]
            last_iter = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print(
                "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                    path, last_iter
                )
            )
            return best_metric, last_iter
    else:
        print("=> no checkpoint found at '{}'".format(path))
