# iris-recognition

----
## How to run this repo

### Clone the repo
```bash
git clone https://github.com/leminhhai2904.git
```
### Download datasets and checkpoint (optionally)
Drive links for checkpoints and datasets
https://drive.google.com/drive/folders/11D7otQcxl9pQbbhH4NukXlX-QTJLxtra?usp=sharing

### Run the file (assumming using pip as package manager)
```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m src.training (or python src/training.py)
```
