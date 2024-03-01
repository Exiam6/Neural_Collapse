Content of this directory:
├── project_directory/
    ├── data/
    ├── configs/
    │   └── configs.py
    ├── utils/
    │   ├── train_utils.py
    │   └── analysis_utils.py
    ├── models/
    │   └── detached_resnet.py
    ├── requirements.txt
    ├── launch.py
    └── train.py

pip install -r requirements.txt

Using parameter parser interactively:
    python launch.py --debug True

To change the figure saving path, modify the fig_saving_pth variable in configs.py.

in train_utils.py, there are:
    def train(model, args, criterion, device, train_loader, optimizer, epoch):
    def set_optimizer(model, args):
    def set_seed(random_seed):
    def load_data(args):

in analysis_utils, there are:
    def analysis(graphs, model, args, criterion_summed, device, loader):