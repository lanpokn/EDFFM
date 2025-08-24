run_project.sh is important, can let you know the environment you need:
source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare

source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare && python main.py --config configs/config.yaml --debug



后续需要完成的：step1还要完成光源事件数据仿真，从flare7K数据集中，读取两个Compound_Flare文件夹中的图片时，实际上在和Compound_Flare文件夹同级的Light_Source文件夹中还有一堆同名的图片。
这些图片就对应了炫光的光源！因此，在step1中，仿真好炫光事件后，还需要用完全一致的频闪与变换曲线应用在光源上图片上，生成光源事件（不加GLSL的反射炫光）然后输出到另一个文件夹lightSource_event中。 请你注意，这个
功能只要光源图片读取以及与炫光配对正常（由于完全重名，炫光图片按照名称的排行与光源图片按照名称的排行应该一致），应该调用几次已有工具就可以简洁的实现，包括光源事件也进行与炫光事件类似的debug可视化。现在请你试着完成这个工作，完成后更新记忆，并告诉我验证你的新的完整step 1的指令（或者你也可以自行验证，给充足的运行时间保证可以运行一组）。