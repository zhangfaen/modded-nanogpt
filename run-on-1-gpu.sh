# total_batch_size == batch_size * sequence_length * os.environ['WORLD_SIZE'] # os.environ['WORLD_SIZE'] is 2, as CUDA_VISIBLE_DEVICES=4,5
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir pylog124M \
    --model d12 \
    --batch_size 64 \
    --sequence_length 1024 \
    --total_batch_size 65536 \
    --val_loss_every 128 \
    --num_iterations 9536 \
    --weight_decay 0.1 \
    --learning_rate 0.0018 \
    --warmup_iters 256 \
    --warmdown_iters 2048


# after run above command, 2 linux processss are running. they are
# (pid:651347) /home/zhangfaen/miniconda3/envs/modded-nanogpt/bin/python /home/zhangfaen/miniconda3/envs/modded-nanogpt/bin/torchrun --standalone --nproc_per_node=1 train_gpt2.py --input_bin data/fineweb10B/fineweb_train_*.bin --input_val_bin data/fineweb10B/fineweb_val_*.bin --output_dir pylog124M --model d12 --batch_size 64 --sequence_length 1024 --total_batch_size 65536 --val_loss_every 128 --num_iterations 9536 --weight_decay 0.1 --learning_rate 0.0018 --warmup_iters 256 --warmdown_iters 2048
# (pid:651414) /home/zhangfaen/miniconda3/envs/modded-nanogpt/bin/python -u train_gpt2.py --input_bin data/fineweb10B/fineweb_train_*.bin --input_val_bin data/fineweb10B/fineweb_val_*.bin --output_dir pylog124M --model d12 --batch_size 64 --sequence_length 1024 --total_batch_size 65536 --val_loss_every 128 --num_iterations 9536 --weight_decay 0.1 --learning_rate 0.0018 --warmup_iters 256 --warmdown_iters 2048

# by run %nvidia-smi, we see pid:651414 are running a gpu.  pid:651347 is not using gpu, it is a master process.


# %cat /proc/651414/environ | tr '\0' '\n'
# LESSOPEN=| /usr/bin/lesspipe %s
# CONDA_PROMPT_MODIFIER=(modded-nanogpt) 
# USER=zhangfaen
# SSH_CLIENT=121.69.59.14 56664 58320
# all_proxy=socks5://192.168.31.40:7890
# XDG_SESSION_TYPE=tty
# SHLVL=1
# LD_LIBRARY_PATH=:/usr/local/cuda/lib64
# MOTD_SHOWN=pam
# HOME=/home/zhangfaen
# OLDPWD=/home/zhangfaen/dev
# LESS=-R
# CONDA_SHLVL=3
# SSH_TTY=/dev/pts/20
# ZSH=/home/zhangfaen/.oh-my-zsh
# LSCOLORS=Gxfxcxdxbxegedabagacad
# PAGER=less
# OLLAMA_HOST=0.0.0.0:11434
# DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/5058/bus
# _CE_M=
# https_proxy=http://192.168.31.40:7890
# CUDA_VISIBLE_DEVICES=4
# LOGNAME=zhangfaen
# http_proxy=http://192.168.31.40:7890
# _=/home/zhangfaen/dev/modded-nanogpt/./run-on-1-gpu.sh
# XDG_SESSION_CLASS=user
# TERM=xterm-256color
# XDG_SESSION_ID=1458
# _CE_CONDA=
# PATH=/home/zhangfaen/miniconda3/envs/modded-nanogpt/bin:/home/zhangfaen/miniconda3/condabin:/usr/local/cuda-12.1/bin:/home/zhangfaen/bin:/usr/local/bin:/home/zfe/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda/bin
# XDG_RUNTIME_DIR=/run/user/5058
# LANG=en_US.UTF-8
# CONDA_PREFIX_1=/home/zhangfaen/miniconda3
# LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.webp=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:
# CONDA_PREFIX_2=/home/zhangfaen/miniconda3/envs/py310
# CONDA_PYTHON_EXE=/home/zhangfaen/miniconda3/bin/python
# SHELL=/bin/bash
# LESSCLOSE=/usr/bin/lesspipe %s %s
# CONDA_DEFAULT_ENV=modded-nanogpt
# PWD=/home/zhangfaen/dev/modded-nanogpt
# CUDA_HOME=/usr/local/cuda-12.1
# SSH_CONNECTION=121.69.59.14 56664 192.168.10.82 58320
# XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
# CONDA_EXE=/home/zhangfaen/miniconda3/bin/conda
# CONDA_PREFIX=/home/zhangfaen/miniconda3/envs/modded-nanogpt
# LOCAL_RANK=0
# RANK=0
# GROUP_RANK=0
# ROLE_RANK=0
# ROLE_NAME=default
# LOCAL_WORLD_SIZE=1
# WORLD_SIZE=1
# GROUP_WORLD_SIZE=1
# ROLE_WORLD_SIZE=1
# MASTER_ADDR=own-jxq-ops-A800-01
# MASTER_PORT=33707
# TORCHELASTIC_RESTART_COUNT=0
# TORCHELASTIC_MAX_RESTARTS=0
# TORCHELASTIC_RUN_ID=82cc045e-a1d1-42cf-959f-0abe79084961
# TORCHELASTIC_USE_AGENT_STORE=False
# TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# TORCHELASTIC_ERROR_FILE=/tmp/torchelastic_hzs73kf3/82cc045e-a1d1-42cf-959f-0abe79084961_mm54nwuq/attempt_0/0/error.json


# %cat /proc/651347/environ | tr '\0' '\n'
# LESSOPEN=| /usr/bin/lesspipe %s
# CONDA_PROMPT_MODIFIER=(modded-nanogpt) 
# USER=zhangfaen
# SSH_CLIENT=121.69.59.14 56664 58320
# all_proxy=socks5://192.168.31.40:7890
# XDG_SESSION_TYPE=tty
# SHLVL=1
# LD_LIBRARY_PATH=:/usr/local/cuda/lib64
# MOTD_SHOWN=pam
# HOME=/home/zhangfaen
# OLDPWD=/home/zhangfaen/dev
# LESS=-R
# CONDA_SHLVL=3
# SSH_TTY=/dev/pts/20
# ZSH=/home/zhangfaen/.oh-my-zsh
# LSCOLORS=Gxfxcxdxbxegedabagacad
# PAGER=less
# OLLAMA_HOST=0.0.0.0:11434
# DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/5058/bus
# _CE_M=
# https_proxy=http://192.168.31.40:7890
# CUDA_VISIBLE_DEVICES=4
# LOGNAME=zhangfaen
# http_proxy=http://192.168.31.40:7890
# _=/home/zhangfaen/dev/modded-nanogpt/./run-on-1-gpu.sh
# XDG_SESSION_CLASS=user
# TERM=xterm-256color
# XDG_SESSION_ID=1458
# _CE_CONDA=
# PATH=/home/zhangfaen/miniconda3/envs/modded-nanogpt/bin:/home/zhangfaen/miniconda3/condabin:/usr/local/cuda-12.1/bin:/home/zhangfaen/bin:/usr/local/bin:/home/zfe/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda/bin
# XDG_RUNTIME_DIR=/run/user/5058
# LANG=en_US.UTF-8
# CONDA_PREFIX_1=/home/zhangfaen/miniconda3
# LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.webp=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:
# CONDA_PREFIX_2=/home/zhangfaen/miniconda3/envs/py310
# CONDA_PYTHON_EXE=/home/zhangfaen/miniconda3/bin/python
# SHELL=/bin/bash
# LESSCLOSE=/usr/bin/lesspipe %s %s
# CONDA_DEFAULT_ENV=modded-nanogpt
# PWD=/home/zhangfaen/dev/modded-nanogpt
# CUDA_HOME=/usr/local/cuda-12.1
# SSH_CONNECTION=121.69.59.14 56664 192.168.10.82 58320
# XDG_DATA_DIRS=/usr/local/share:/usr/share:/var/lib/snapd/desktop
# CONDA_EXE=/home/zhangfaen/miniconda3/bin/conda
# CONDA_PREFIX=/home/zhangfaen/miniconda3/envs/modded-nanogpt


# 
# %ls -l /proc/651414/fd/1 /proc/651414/fd/2
# lrwx------ 1 zhangfaen guanli 64 Jun 10 14:43 /proc/651414/fd/1 -> /dev/pts/20
# lrwx------ 1 zhangfaen guanli 64 Jun 10 14:43 /proc/651414/fd/2 -> /dev/pts/20
# 
# 
# %ls -l /proc/651347/fd/1 /proc/651347/fd/2
# lrwx------ 1 zhangfaen guanli 64 Jun 10 15:01 /proc/651347/fd/1 -> /dev/pts/20
# lrwx------ 1 zhangfaen guanli 64 Jun 10 15:01 /proc/651347/fd/2 -> /dev/pts/20
# 
# %ps -t pts/20
#     PID TTY          TIME CMD
#  643602 pts/20   00:00:00 zsh
#  651346 pts/20   00:00:00 sh
#  651347 pts/20   00:00:08 pt_main_thread
# 
#  %ps -ef --forest | grep 651414
# zhangfa+  652404  643634  0 15:02 pts/22   00:00:00  |           \_ grep --color=auto --exclude-dir=.bzr --exclude-dir=CVS --exclude-dir=.git --exclude-dir=.hg --exclude-dir=.svn --exclude-dir=.idea --exclude-dir=.tox 651414
# zhangfa+  651414  651347 99 14:43 ?        00:48:31  |                   \_ /home/zhangfaen/miniconda3/envs/modded-nanogpt/bin/python -u train_gpt2.py --input_bin data/fineweb10B/fineweb_train_*.bin --input_val_bin data/fineweb10B/fineweb_val_*.bin --output_dir pylog124M --model d12 --batch_size 64 --sequence_length 1024 --total_batch_size 65536 --val_loss_every 128 --num_iterations 9536 --weight_decay 0.1 --learning_rate 0.0018 --warmup_iters 256 --warmdown_iters 2048