CUDA_VISIBLE_DEVICES=0 python train_sr_dr.py --overlap_ratio 0.25 -ds "mybank" -dm "loan_account" --model "sasrec" --overlap True --isItC True --ts2 0.4 --neg_nums 999 --lr2 0.01 --dr_e_w 0.01 