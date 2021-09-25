# A Guide to set up PyTorch 1.9.1 w. Python to use within PyCharm on your Windows machine

## Disclaimer, the entire setup takes quite some time - partially depending on your internet and HW speed! Furthermore, this setup has been tested with an NVIDIA CUDA Enabled GPU in place

### 0. Download and Install CUDA Toolkit and cuDNN from NVIDIA
Note: I work with cuda 11.4.2.471.41 in this tutorial

[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

Note: You need an NVIDIA Account to download the following (I work with cudnn-11.0-windows-x64-v8.0.5.39 in this tutorial)

[cuDNN](https://developer.nvidia.com/cudnn)

### 1. Download and install Anaconda or mini-conda (lightweight Anaconda without UI)
[Download Anaconda](https://www.anaconda.com/products/individual)

### 2. Download & install PyCharm Community
[Download PyCharm Community](https://www.jetbrains.com/de-de/pycharm/download/#section=windows)

### 3.
## The following instruction part is taken from Jeff Heaton's [t81_558_deep_learning class](https://github.com/jeffheaton/t81_558_deep_learning), all credits go out to him! I have slightly adjusted it to my needs and to the currently available PyTorch version

## NOTE

## You will also find a YT tutorial video where he guides you through his setup [Jeff Heaton - PyTorch Setup](https://www.youtube.com/watch?v=vBfM5l9VK5c)
<a href="https://colab.research.google.com/github/jeffheaton/t81_558_deep_learning/blob/master/manual_setup.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# T81-558: Applications of Deep Neural Networks
**Manual Python Setup**
* Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
* For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# Software Installation

Note, this class is (curretnly) primarily focused on TensorFlow/Keras; however, I am providing some examples of PyTorch.  This notebook described how to install PyTorch for either GPU or CPU.

## Installing Python and PyTorch

It is possible to install and run Python/PyTorch entirely from your computer, without the need for Google CoLab. Running PyTorch locally does require some software configuration and installation.  If you are not confortable with software installation, just use Google CoLab.  These instructions show you how to install PyTorch for both CPU and GPU. Many of the examples in this class will achieve considerable performance improvement from a GPU.


The first step is to install Python 3.8.  I recommend using the Miniconda (Anaconda) release of Python, as it already includes many of the data science related packages that are needed by this class.  Anaconda directly supports Windows, Mac, and Linux.  Miniconda is the minimal set of features from the extensive Anaconda Python distribution.  Download Miniconda from the following URL:

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

First, lets install Jupyter, which is the editor you will use in this course.

```
conda install -y jupyter
```

We will actually launch Jupyter later.

You must make sure that PyTorch has the version of Python that it is compatible with.  The best way to accomplish this is with an Anaconda environment.  Each environment that you create can have its own Python version, drivers, and Python libraries.  I suggest that you create an environment to hold the Python instance for this class.  Use the following command to create your environment. I am calling the environment **torch**, you can name yours whatever you like.

```
conda create --name torch python=3.8
```

To enter this environment, you must use the following command: 

```
conda activate torch
```


For now, lets add Jupyter support to your new environment.

```
conda install nb_conda
```

We will now install PyTorch.  We will make use of conda for this installation. The next two sections describe how to install PyTorch for either a CPU or GPU. To use GPU, you must have a [compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus).


## Install PyTorch for CPU Only
The following command installs PyTorch for CPU support.  Even if you have a GPU, it will not be used.

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```


## Install PyTorch for GPU and CPU

The following command installs PyTorch for GPU support.  All of the complex driver installations should be handled by this command.

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

## Install Additional Libraries for ML

There are several additional libraries that you will need for this course.  This command will install them.  Make sure you are still in your **pytorch** environment.

```
conda env update --file tools.yml
```

The file bears the following content:

```
dependencies:
    - jupyter
    - scikit-learn
    - scipy
    - pandas
    - pandas-datareader
    - matplotlib
    - pillow
    - tqdm
    - requests
    - h5py
    - pyyaml
    - flask
    - boto3
    - pip
    - pip:
        - bayesian-optimization
        - gym
        - kaggle
```

The [tools.yml](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/tools.yml) file is located in the root directory for this GitHub repository.

## Register your Environment

The following command registers your **pytorch** environment. Again, make sure you "conda activate" your new **pytorch** environment.

```
python -m ipykernel install --user --name pytorch --display-name "Python 3.8 (pytorch)"
```

## Testing your Environment

You can now start Jupyter notebook.  Use the following command.

```
jupyter notebook
```

You can now run the following code to check that you have the versions expected.


```python
# What version of Python do you have?
import sys

import torch
import pandas as pd
import sklearn as sk

print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if torch.cuda.is_available() else "NOT AVAILABLE")
```

    PyTorch Version: 1.9.1
    
    Python 3.8.11 (default, Aug  6 2021, 09:57:55) [MSC v.1916 64 bit (AMD64)]
    Pandas 1.3.2
    Scikit-Learn 0.24.2
    GPU is available
    
### 4. Integration with PyCharm

4.1. Open PyCharm, and navigate to: File > Settings &#8594; enter
![Settings](/img/00%20-%20PyCharm%20-%20Settings.jpg)

4.2. Navigate within Settings to: Project:"whatever name you chose" > Python Interpreter, click on the gear wheel right to the Python Interpreter selection, and click on "Show All..."

![Python Interpreter](/img/01%20-%20PyCharm%20-%20Python%20Interpreter.jpg)
![Python Interpreter Show all](/img/02%20-%20PyCharm%20-%20Python%20Interpreter%20-%20Show%20all.jpg)

4.3. Click on the "add" symbol

![Python Interpreter Add](/img/03%20-%20PyCharm%20-%20Python%20Interpreters%20-%20add.jpg)

4.4. Select "Conda Environment"

![Conda Environment](/img/04%20-%20PyCharm%20-%20Python%20Interpreters%20-%20Conda%20Env.jpg)

4.5. Choose "Existing environment", pick the environment "torch", which you have created prior in this instruction, and press OK

![Conda Environment](/img/05%20-%20PyCharm%20-%20Python%20Interpreters%20-%20Conda%20Torch.jpg)

4.6. You should see the Python Interpreter now in the selection, if so, press OK

![Conda Environment](/img/06%20-%20PyCharm%20-%20Python%20Interpreters%20-%20Torch%20-%20Accept.jpg)

Congrats, the setup is now done, it should now be possible to run PyTorch from within PyCharm.

### 5. (OPTIONAL) Test CUDA Cores
If you have a CUDA enabled NVIDIA Graphics Card you may want to test, whether PyTorch is actually running on your GPU instead of CPU. Without specification PyTorch will use your CPU.

5.1 For testing purposes I have taken a little implementation done by "Python Engineer", you will find his instruciton video here: [PyTorch Tutorial 07 - Linear Regression](https://www.youtube.com/watch?v=YAJ5XBwlN4o)

However, in his implementation the code runs on the CPU, in order to make it run on the GPU one needs to add ".cuda()" to the model and to the tensors. To save you some time, you can simply open the "linear-regression.py" file in PyCharm and run it.

5.2 You should see some output like this in your terminal:

![PyCharm - Result](/img/07%20-%20PyCharm%20-%20Result.jpg)

But how do you know whether this has been actually running on your GPU?

5.3 Check yur "Task Manager"
Open your Task Manager by pressing `crtl + alt + del` and choose `Task Manager`, then navigate to the `Performance` tab and choose `GPU`

![Task MGR - GPU](/img/08.1%20-%20TaskMgr%20-%20GPU%20-%20pick.jpg)

Then change one of the monitors to `CUDA`

![Task MGR - CUDA](/img/08%20-%20TaskMgr%20-%20GPU%20-%20CUDA.jpg)

5.4 Re-run `linear-regression.py` and check back whether the monitor shows utilization of CUDA, if so, you are running that model on your GPU instead of CPU

![Task MGR - Utilization](/img/09%20-%20TaskMgr%20-%20GPU%20-%20CUDA%20-%20Utilization.jpg)

### 6. (OPTIONAl) Conda TensorFlow + PyCharm Integration
I have followed the tutorial in this video [Installing Latest TensorFlow](https://www.youtube.com/watch?v=hHWkvEcDBO0) to make TF run on my local machine.
Some of the steps described above, will also apply for activation of TF in Pycharm.