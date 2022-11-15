# Light-MALib

[![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](./LICENSE)
[![Release Version](https://img.shields.io/badge/release-0.1.0-red.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

This repo mainly provides a **simplified** version of [**MALib**](https://github.com/sjtu-marl/malib) codes with restricted algorithms but also certain enhancements like distributed async-training, league-like multiple population training, detailed tensorboard logging.

Currently, it is **dedicated** for **Google Research Football** environment. <u>In the future, we will also release codes for other algorithms and environments</u>.

Our codes are designed to be easy to modify, therefore relatively **light-weighted** compared to the original delicately-designed [**MALib**](https://github.com/sjtu-marl/malib) codes, which have full support for the study of cutting-edge Multi-Agent RL algorithms. So we name our repo **Light-MALib**.

If you have further needs beyond this repo, please refer to [**MALib**](https://github.com/sjtu-marl/malib) for more support.

## Contents
1. Install
2. Run Experiments
3. Contact

## Install
You can use any tool to manage your python environment. Here, we use conda as an example.
1. install conda/minconda.
2. `conda create -n light-malib python==3.8` to create a new conda env.
3. activate the env by `conda activate light-malib` when you want to use it or you can add this line to your `.bashrc` file to enable it everytime you login into the bash.

### Install Light-MALib, PyTorch and Google Research Football
1. In the root folder of this repo (with the `setup.py` file), run `pip install -r requirement.txt` to install dependencies of Light-MALib.
2. In the root folder of this repo (with the `setup.py` file), run `pip install .` or `pip install -e .` to install Light-MALib.
3. Follow the instructions in the official website https://pytorch.org/get-started/locally/ to install PyTorch (for example, version 1.13.0+cu116).
4. Follow the instructions in the official repo https://github.com/google-research/football and install the Google Research Football environment.

### Add a New Football Game Scenario
1. You may use `python -c "import gfootball;print(gfootball.__file__)"` or other methods to locate where `gfootball` pacakage is. 
2. Go to the directory of `gfootball` pacakage, for example, `/home/username/miniconda3/envs/light-malib/lib/python3.8/site-packages/gfootball/`.
3. Copy `.py` files under `scenarios` folder in our repo to `scenarios` folder in the `gfootball` pacakage.

## Run Experiments
1. If you want to run experiments on a small cluster, please follow [ray](https://docs.ray.io/en/latest/ray-core/starting-ray.html)'s official instructions to start a cluster. For example, use `ray start --head` on the master, then connect other machines to the master following the hints from command line output.
2. `python light_malib/main_pbt.py --config <config_file_path>` to run a training experiment. An example is given by `train_light_malib.sh`.
3. `python light_malib/scripts/play_gr_football.py` to run a competition between two models. 

## Contact
If you have any questions about this repo, feel free to leave an issue. You can also contact current maintainers, [YanSong97](https://github.com/YanSong97) and [DiligentPanda](https://github.com/DiligentPanda), by email.
