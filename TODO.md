run_project.sh is important, can let you know the environment you need:
source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare

source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare && python main.py --config configs/config.yaml --debug


禁用val，或者val从其他地方拿，不然本来也是仿真的东西
保存的h5重名问题
flare的频率可以再翻倍
存的是处理过的x,y,dt,p和label而不是原始的xytp,这点要注意，后续想补feature extractor就不容易了

面向val_loss来调整学习率