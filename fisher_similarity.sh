nohup python -u Fisher_information_experiment.py --device=0 --model='Resnet' > ./Fisher_information/resnet18.txt &
nohup python -u Fisher_information_experiment.py --device=0 --model='densenet' > ./Fisher_information/densenet161.txt &
nohup python -u Fisher_information_experiment.py --device=0 --model='vgg16' > ./Fisher_information/vgg16.txt &
