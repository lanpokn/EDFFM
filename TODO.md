run_project.sh is important, can let you know the environment you need:
source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare

source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare && python main.py --config configs/config.yaml --debug


禁用val，或者val从其他地方拿，不然本来也是仿真的东西
保存的h5重名问题
flare的频率可以再翻倍
存的是处理过的x,y,dt,p和label而不是原始的xytp,这点要注意，后续想补feature extractor就不容易了

面向val_loss来调整学习率

inference怎么这么慢？有余力去思考实时推理

服务器上的正确流程：

好的，我们经历了漫长而曲折的调试过程，最终克服了环境、网络、编译和依赖的重重障碍。现在，我将总结并提炼出经过您反复验证和修正后的最终正确、完整的服务器部署全流程。

这个流程是为您特定的服务器环境（Linux、无管理员权限、网络连接GitHub和PyPI不稳定、系统CUDA 11.8、新版g++）量身定制的。

服务器端部署终极指南 (从零到成功)
阶段一：环境准备 (Environment Setup)

此阶段的目标是创建一个干净、版本精确且包含所有编译工具的Conda环境。

创建并激活Conda环境

code
Bash
download
content_copy
expand_less

# 创建一个名为 event_flare 的独立环境
conda create -n event_flare python=3.10 -y

# 激活环境，之后所有操作都在此环境中进行
conda activate event_flare

安装核心依赖：精确版本的PyTorch与CUDA
这是最关键的一步，确保PyTorch版本与后续的Mamba版本兼容，且CUDA版本与您服务器的物理环境（nvcc --version 显示 11.8）一致。

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# 安装PyTorch 2.2.1的CUDA 11.8版本
conda install pytorch==2.2.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

安装兼容的C++编译器
为解决CUDA 11.8不支持高版本g++的问题，我们手动安装一个兼容的g++ 11.x版本。

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# 安装版本号为11.x的g++编译器套件
conda install -c conda-forge "gxx_linux-64==11.*"
阶段二：手动编译安装核心组件

此阶段的目标是绕过所有不稳定的网络下载和有问题的pip自动化逻辑，通过手动下载、本地编译的方式安装causal-conv1d和mamba-ssm。

准备源码（在本地电脑操作）

步骤a： 在您的本地电脑（新加坡）上，通过浏览器下载以下两个源码压缩包：

causal-conv1d v1.1.0: https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.1.0.zip

mamba v1.1.0: https://github.com/state-spaces/mamba/archive/refs/tags/v1.1.0.zip

步骤b： 使用百度网盘或其他可靠方式，将这两个.zip文件上传到您的服务器，例如放在~/eventFlare/目录下。

在服务器上编译并“手动安装”causal-conv1d

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# 进入工作目录
cd ~/eventFlare/

# 解压缩源码包
unzip causal-conv1d-1.1.0.zip

# 进入源码目录
cd causal-conv1d-1.1.0

# 执行纯本地编译 (这步将生成.so文件)
python setup.py build_ext --inplace

# 手动创建符号链接完成“安装”
ln -s ~/eventFlare/causal-conv1d-1.1.0/causal_conv1d ~/miniconda3/envs/event_flare/lib/python3.10/site-packages/causal_conv1d

在服务器上编译并“手动安装”mamba-ssm

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# 回到工作目录
cd ~/eventFlare/

# 解压缩源码包
unzip mamba-1.1.0.zip

# 进入源码目录
cd mamba-1.1.0

# 执行纯本地编译 (这步将生成selective_scan_cuda.so等文件)
python setup.py build_ext --inplace

# 手动创建符号链接完成“安装”
ln -s ~/eventFlare/mamba-1.1.0/mamba_ssm ~/miniconda3/envs/event_flare/lib/python3.10/site-packages/mamba_ssm
阶段三：安装剩余依赖包

此阶段安装所有剩余的、纯Python的依赖包。

安装ninja和einops
这两个是Mamba运行时的直接依赖。

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
pip install ninja einops

安装requirements.txt中的其他项目依赖

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# 进入您的项目主代码目录
cd ~/eventFlare/main

# 安装requirements.txt
pip install -r requirements.txt

处理NumPy和OpenCV的兼容性问题
pip在安装requirements.txt时可能会引入不兼容的NumPy 2.x和依赖它的OpenCV。我们最后用Conda强制修正这个问题。

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# 使用Conda安装一个与PyTorch兼容的NumPy 1.x版本，以及一个同样兼容的OpenCV版本
conda install "numpy<2" opencv
阶段四：验证与运行

最终验证
在任何目录下运行此命令，检查核心组件是否都已正确安装并可以导入。

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
python -c "import torch; import causal_conv1d; import mamba_ssm; print('✅ All dependencies seem OK!')"

处理配置文件格式（如需要）
如果您的.yaml配置文件是从Windows系统创建的，请在运行前转换为Unix格式。

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# (可选) 用Conda安装dos2unix工具
# conda install -c conda-forge dos2unix

# 转换文件
dos2unix your_config_file.yaml

使用tmux安全地运行您的任务

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# 启动一个名为 my_job 的新会话
tmux new -s my_job

# 在tmux会话中，激活环境并运行您的代码
conda activate event_flare
cd ~/eventFlare/main
python your_training_script.py --config configs/config.yaml

# 按 Ctrl+b 然后 d 来分离会话，让任务在后台继续运行

这个流程凝聚了我们所有的调试经验，每一步都是为了解决一个您实际遇到过的问题。它应该可以作为一个可靠的指南，帮助您或其他人在类似的环境下顺利完成部署。