
def find(args, arg):
    """
    查找args中是否有指定的arg，若存在则返回arg对应的数据
    -p aaa
    所以返回 args[i + 1], ie. aaa
    """
    try:
        i = args.index(arg)
        return args[i + 1]
    except:
        raise Exception(f"Missing required argument: {arg}")


def parse_args(args):
    # args[0]为 py文件，所以取 args[1:]及其以后的数据
    args = args[1:]

    config_dict = {}
    # d--dataset    p--program  i--bug_id   m--method ie.FL method  e--exp ie.resample or ConDiffusion
    required_args = ["-d", "-p", "-i", "-m", "-e"]
    for arg in required_args:
        config_dict[arg] = find(args, arg)  # args[args.index(arg) + 1]

    if ((config_dict["-e"] == "origin") or (config_dict["-e"] == "resampling") or (
            config_dict["-e"] == "undersampling") or (config_dict["-e"] == "cvae")) and len(config_dict) != 5:
        raise Exception(f"{config_dict['-e']} has no -cp or -ep")

    # fs means "feature selection"
    # 降维 + 数据增强
    if config_dict["-e"] == "fs" or config_dict["-e"] == "fs_cvae" or config_dict["-e"] == "fs_ddpm" or config_dict["-e"] == "lda_smote":
        config_dict["-cp"] = find(args, "-cp")
        config_dict["-ep"] = find(args, "-ep")


    # 最终判断
    if config_dict["-e"] not in ["origin", "resampling", "undersampling", "fs", "cvae", "fs_cvae", "smote", "fs_ddpm", "lda_smote", "slice_mixup", "fs_cgan"]:
        raise Exception(f"Wrong parameters {config_dict}, please check again.")

    optional_args = ["-r", "-a"]
    for arg in optional_args:
        if arg in args:
            config_dict[arg] = args[args.index(arg) + 1]

    return config_dict
