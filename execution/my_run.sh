cd ../src

#../parallel -j18 --resume-failed --results ../Output/midtertm2 --joblog ../joblog/midtertm2  CUDA_VISIBLE_DEVICES=1 python ./sparse_Magnet.py  --epochs 3000 --lr {1} --num_filter {2} --q {3} --log_path midtertm2 --dataset WebKB/Wisconsin  --K 1  --layer 2 --dropout 0.5 -a ::: 0.001 0.01 0.005 ::: 16 32 64 ::: 0.0 0.05 0.1 0.15 0.2 0.25

#../parallel -j4 --resume-failed --results ../Output/LinkPred1    --joblog ../joblog/LinkPred1    CUDA_VISIBLE_DEVICES=0 python ./Edge_sparseMagnet.py --epochs 3000 --num_filter {1} --dataset {2} --q {3} --K=1 --lr 1e-3 --split_prob=0.15,0.05  --dropout 0.5  --task 1 --log_path LinkPred1 ::: 16 32 64 :::  WebKB/Wisconsin   ::: 0.05 0.1 0.15 0.2 0.25

python ./sparse_Magnet.py  --epochs 3000 --lr 0.005 --num_filter 16 --q 0.1 --log_path try_8_930  --data_path dataset/ --dataset spatial/data_8_930.csv  --K 1  --layer 2 --dropout 0.5 -a 