Build on top of wsl Ubuntu 24.4

Installation : 

Enables wsl features 
```wsl --intall```

Install a distro (Ubuntu), some distro can be found in the microsoft store, Ubuntu 24.4 was used for this repo

Enables wsl 2 
```
wsl --list --verbose # Check current version
wsl --set-version Ubuntu-24.04 2 # If VERSION shows 1, convert to 2
```

Shut down wsl then restart wsl
```
wsl --shutdown
wsl
```

Install python3, pip and venv 
```
apt update
apt install -y python3 python3-pip python3-venv
```

Create virtual env in the target folder:
```
python3 -m venv venv
```

Install the required packages 
```
pip install -r requirements.txt
```

Install OpenGL libraries (for rendering during testing)
```
apt install -y libglu1-mesa libglu1-mesa-dev mesa-utils libgl1-mesa-dri
```

