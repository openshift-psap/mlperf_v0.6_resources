python ./qa/testscript.py /data/imagenet --data-backends pytorch dali-gpu dali-cpu --bs 224 256 448 512 --ngpus 16 --workspace $1 -j 3 --arch resnet50 -c fanin --label-smoothing 0.1 --amp --static-loss-scale 128 --mode inference --mixup 0.0