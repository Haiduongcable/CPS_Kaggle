import os 
import numpy as np 

path_resnet = "/home/asilla/duongnh/project/CrossPseudo_UpdateBranch/CPS_Kaggle/Log_result/Log_resnet101_color"
path_contrastive = "/home/asilla/duongnh/project/CrossPseudo_UpdateBranch/CPS_Kaggle/Log_result/Log_resnet101_contrastive_color"

dict_score = {}
l_con = sorted(os.listdir(path_contrastive))
l_resnet = sorted(os.listdir(path_resnet))
for index, nimage in enumerate(l_resnet):

    nimage_con = l_con[index]
    print(nimage_con, nimage)
    score_resnet = float(nimage.split("_")[-1][:-4])
    score_con = float(nimage_con.split("_")[-1][:-4])
    compare_score = score_con - score_resnet
    convert_n_image = nimage.split("_")[0] + "_" + nimage.split("_")[1] + ".png" 
    dict_score[convert_n_image] = compare_score

sorted_dict_score = {k: v for k, v in sorted(dict_score.items(), key=lambda item: item[1])}
print(sorted_dict_score)