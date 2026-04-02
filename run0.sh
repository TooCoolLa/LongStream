python -u run.py  --img-path data/ --seq-list 00 --checkpoint "checkpoints/50_longstream.pt" --streaming-mode window --window-size 100 --keyframe-stride 1 --refresh 110 2>&1 | tee infer.log
