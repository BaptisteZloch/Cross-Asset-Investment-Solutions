# Project for the course Cross Asset Investment Solution with Paris-Dauphine University.
The aim is to create a cross asset investment convex solution for a corporate client.
# Setting up the project
Run the file `install_for_windows.bat` it will install dependencies and create a virtual environment for the project.
Or you can launch the following commands in your terminal : 
- Windows
```batch
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install -U pandas
```
- Linux/MacOS
```shell
python -m venv .venv # or python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip  install -U pandas
```
# Main work location
All the code is in the `src` folder. The main analysis is located at `src\backtest_portoflio.ipynb`.<br>

The project guidelines are in the `static` folder.