run_project.sh is important, can let you know the environment you need:
source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare

source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare && python main.py --config configs/config.yaml --debug


禁用val，或者val从其他地方拿，不然本来也是仿真的东西
保存的h5重名问题
flare的频率可以再翻倍
存的是处理过的x,y,dt,p和label而不是原始的xytp,这点要注意，后续想补feature extractor就不容易了

面向val_loss来调整学习率？

inference怎么这么慢？有余力去思考实时推理.

目前去炫光效果并不好，但也许只是训练还没有收敛？

重新引入PFD，让gemeni想该怎么做

现在的inference并不对，数据集也不是原始数据，肯定不够好，都需要改动。不过我现在最需要的恐怕不是沉浸在这些无聊的工程细节上，而是想好一种最基本的
数据生成应该有各种不同的mode