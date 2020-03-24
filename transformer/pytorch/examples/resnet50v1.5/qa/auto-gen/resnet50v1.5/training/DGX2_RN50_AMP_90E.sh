python ./multiproc.py --nproc_per_node 16 ./main.py --raport-file raport.json -j5 -p 100 --lr 4.096 --optimizer-batch-size 4096 --warmup 16 --arch resnet50 -c fanin --label-smoothing 0.1 --data-backend pytorch --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace $1 -b 256 --amp --static-loss-scale 128 --epochs 90 /data/imagenet