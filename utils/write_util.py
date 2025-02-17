import os
import pandas as pd


def write_corr_to_txt(method, corr_dict, path, state):

    corr_dict = pd.DataFrame.from_dict(corr_dict, orient='index', columns=['susp'], dtype=float)
    corr_dict = corr_dict.reset_index().rename(columns={'index': 'line_num'})
    corr_dict['line_num'] = corr_dict['line_num'].astype(int)
    corr_dict.sort_values(by=['susp', 'line_num'], ascending=[False, True], inplace=True)

    # path--xxx/results     state---exp
    save_path = os.path.join(path, state)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_file_name = method + "-" + state + ".txt"
    concrete_path = os.path.join(save_path, res_file_name)
    # "D:\university_study\科研\Diffusion_Model\d4j\Chart\1\fs_ddpm\dstar-fs_ddpm.txt"

    with open(concrete_path, 'w') as f:
        for each in corr_dict.values:
            print(str(int(each[0])) + "  " + str(each[1]), file=f)


def write_rank_to_txt(rank_dict, sava_path, program, bug_id):
    with open(sava_path, 'a') as f:
        value = f"{program}-{bug_id}\t\t"
        for v in rank_dict.values():
            value += str(v) + "\t"
        print(value, file=f)
