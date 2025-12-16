Stable retro needs to be called inside a wsl when working on windows, this code was built on top of wsl Ubuntu 24.04
installation : 

Enables wsl features 
```
wsl --intall
```

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

***Code Documentation --------------***

