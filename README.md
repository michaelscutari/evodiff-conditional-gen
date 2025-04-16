# Installation
1. Install [EvoDiff](https://github.com/microsoft/evodiff) according to their instructions. They recommend using `python 3.8`.
2. To get clustering to work with `protclust`, you will need to have `mmseqs2` installed in the command line. If you are using conda or micromamba, this is as simple as running `conda install -c bioconda mmseqs2`. You can visit the [mmseqs2](https://github.com/soedinglab/MMseqs2) GitHub for more
installation options.

# Data Collection
1. Begin with scripts in the `preprocessing` directory, starting with `download_data.py`. By altering the `main` function, you can change which EC class you want to download. If you are starting, I recommend using *EC 5* since it is the smallest.
2. Next, access `filter.ipynb`. You should modify the `pandas` at the beginning to import your data, wherever you stored it. From here, the notebook will combine the data, assign it labels, and cluster it using `protclust`.

# Training
At this point, you are ready to run. Try running `train.py` making sure that the data directory correctly points to your stored data. `train_full.py` implements several useful features like `Autocast`, learning rate warmup and scheduling, and advanced logging.
