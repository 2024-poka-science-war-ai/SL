import os
def get_epoch():
    ckpts = os.listdir("./models")
    ckpt_dict = dict()
    for ckpt in ckpts:
        if not ckpt.startswith("test1") or not ckpt.endswith(".pt"):
            continue
        print(ckpt)
        epoch = int(ckpt.split("_")[1][5:])
        if epoch in ckpt_dict.keys():
            ckpt_dict[epoch].append(ckpt)
        else:
            ckpt_dict[epoch] = [ckpt]

    return ckpt_dict

print(get_epoch())

