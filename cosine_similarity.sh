nohup python -u Cosine_90_degrees_resnet18.py --device=0  > ./cosine_transfer_record/resnet18.txt &
nohup python -u Cosine_90_degrees_densenet.py --device=0  > ./cosine_transfer_record/densenet161.txt &
nohup python -u Cosine_90_degrees_vgg16.py --device=0  > ./cosine_transfer_record/vgg16.txt &
