# Setup and Run Guide for InceptSADNet

This guide explains how to set up the environment, prepare the data, and run the `InceptSADNet` model on your local machine with a GV100 GPU.

## 1. Environment Setup

### Prerequisites
- Anaconda or Miniconda installed.
- CUDA drivers installed (compatible with your GV100).

> [!WARNING]
> **Disk Space**: If your C: drive is full, you can configure Conda to use D: drive:
> ```bash
> # 1. Create directories on D:
> mkdir "D:\conda_envs"
> mkdir "D:\conda_pkgs"
>
> # 2. Add them to conda config
> conda config --add envs_dirs "D:\conda_envs"
> conda config --add pkgs_dirs "D:\conda_pkgs"
> ```

### Create Conda Environment
Open your terminal (Anaconda Prompt or PowerShell with conda initialized) and run the following commands:

1.  Create a new environment name 'dsadnet' with Python 3.8:
    ```bash
    conda create -n dsadnet python=3.8 -y
    ```

2.  Activate the environment:
    ```bash
    conda activate dsadnet
    ```

3.  Install PyTorch with CUDA 11.8 support (Recommended for GV100):
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    ```

4.  Install other dependencies:
    ```bash
    pip install -r requirements.txt
    ```

**Note:** If you encounter issues with `mne` or `scikit-learn`, ensure they are installed correctly via pip or conda.

## 2. Data Preparation

The repository requires a specific data structure in `data/raw`. We have provided a script `setup_data.py` to automatically organize your dataset from `D:\fullSADT`.

1.  Make sure your zip files are in `D:\fullSADT`.
2.  Run the setup script:

    ```bash
    python setup_data.py
    ```

This script will:
- Read zip files from `D:\fullSADT`.
- Extract `.set` and `.fdt` files.
- Place them into `d:\DSADNet-main\data\raw\sXX\` folders (e.g., `s01`, `s02`), flattening any internal directory structure.

3.  **Generate Preprocessed Data**:
    The model needs `.npy` files which are generated from the raw `.set` files.
    
    ```bash
    python make_datasets.py
    ```
    
    This will take some time as it processes all subjects and saves `_x.npy` and `_y.npy` files in the subject directories.

## 3. Running the Model

Once data is prepared, you can run the model.

### Training Logic
The main entry point is `run.py`. We have updated it to use the `InceptSADNet` model configuration.

To run the training:

```bash
python run.py --model InceptSADNet
```

### Configuration
You can modify `models/InceptSADNet.py` to change hyperparameters:
- `learning_rate`
- `batch_size`
- `num_epoch`

By default, it uses `data/raw` which we populated in step 2.

### Output
- Logs will be saved in `data/log/InceptSADNet`.
- Model checkpoints (f1/auc) will be saved in `data/saved_dict/`.

## 4. Troubleshooting

- **Import Errors**: Ensure you are in the `dsadnet` environment.
- **Data Not Found**: Check `data/raw` to see if `sXX` folders exist and contain `.set` files.
- **CUDA/GPU issues**: Run `python -c "import torch; print(torch.cuda.is_available())"` to verify GPU access.

### Conda Activation Issues
If `conda activate dsadnet` fails with "Run conda init", try **closing and reopening your terminal**. 
If that doesn't work, you can use the absolute path to the Python executable directly:

```bash
# Run setup script
D:\conda_envs\dsadnet\python.exe setup_data.py

# Install requirements
D:\conda_envs\dsadnet\python.exe -m pip install -r requirements.txt

# Run the model
D:\conda_envs\dsadnet\python.exe run.py --model InceptSADNet
```

### Pip "No space left on device"
If `pip install` fails with "No space left on device", it means your C: drive temp folder is full. 
Run these commands in PowerShell to use D: drive for temporary files:

```powershell
# 1. Create a temp folder on D:
mkdir "D:\tmp"

# 2. Set environment variables (for this session only)
$env:TMP = "D:\tmp"
$env:TEMP = "D:\tmp"
$env:PIP_CACHE_DIR = "D:\pip_cache"

# 3. Run pip install again
D:\conda_envs\dsadnet\python.exe -m pip install -r requirements.txt --no-cache-dir
```

### WandB API Key
The code now prompts for your Weights & Biases API Key.
1.  Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize) or your settings page.
2.  When you run the code, paste it when prompted.
3.  Alternatively, set it as an environment variable:
    ```powershell
    $env:WANDB_API_KEY = "your_api_key_here"
    ```
<!-- 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 -->