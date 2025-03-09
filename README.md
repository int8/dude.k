# dude-k 

This repository contains code for training and evaluating slightly refined TDEED model for 2025 Ball Action Spotting Challenge. 
The details of the challenge can be found here: https://www.soccer-net.org/tasks/ball-action-spotting 


Contribution: 

- refactor of the starter kit code to be a bit more pleasant to work with
- Usage of TDEED model (winner from previous year) https://github.com/SoccerNet/sn-teamspotting/tree/main - my implementation get rid of team head and uses 2x more labels instead 
- Usage of augmentation ideas from https://github.com/lRomul/ball-action-spotting/tree/master 
- Pretraining on broadcast videos 
- Fine-tuning on SN-BAS-2025 dataset 
- Hyperparameter tuning on SN-BAS-2025 
- bunch of scripts to extract frames / pretraining / training / submission


## Installation

```bash
uv sync
uv pip install -e .  
```
---

## Datasets
To get started you need to download SoccerNet datasets

---


### SN-BAS-2025

https://huggingface.co/datasets/SoccerNet/SN-BAS-2025 

First download the dataset from huggingface data hub:
```python 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="SoccerNet/SN-BAS-2025",
                  repo_type="dataset", revision="main",
                  local_dir="SoccerNet/SN-BAS-2025")
```


After signing NDA and receiving password you go to `SoccerNet/SN-BAS-2025` and unzip 
`train.zip` `test.zip` and `valid.zip` files into **the same** directory (you basically ignore the existing splits)


```bash
#  7-zip example
7z x train.zip -o/path/to/data/soccernet/sn_bas_2025
7z x test.zip -o/path/to/data/soccernet/sn_bas_2025
7z x valid.zip -o/path/to/data/soccernet/sn_bas_2025
```

(`7z` will ask you for the password you receive via email after signing NDA)

---

### Broadcast videos (Optional)


It is recommended to download dataset from challenge from previous years. They have some overlapping/similar labels and will be useful for pretraining.
Be aware that this dataset is pretty beefy - it contains around `1.2TB` of videos.


https://www.soccer-net.org/data#h.ov9k48lcih5g

```python 
from SoccerNet.Downloader import SoccerNetDownloader
downloader=SoccerNetDownloader(LocalDirectory="/path/to/data/soccernet/videos")
downloader.password = "Password for videos (received after filling the NDA)"
downloader.downloadGames(
    files=["1_720p.mkv", "2_720p.mkv"],
    split=["train","valid","test",]
)
downloader.downloadGames(
    files=["1_224p.mkv", "2_224p.mkv"], 
    split=["train","valid","test"]
)
```

(it may happen that transfer drops unexpectedly - the workaround for it is to use this code: https://gist.github.com/int8/6b5db0e6e16dfaa8bea9154d3774dec3) 

At this point `/path/to/data/soccernet/videos` contains all the historical broadcasts. 

Broadcast videos themselves do not contain any annotations. They are included in `SoccerNet/SN-BAS-2025/ExtraLabelsActionSpotting500games` you downloaded from huggingface data hub earlier.

 unzip them into `/path/to/data/soccernet/videos` 
```shell
7z x train_labels.zip -o/path/to/data/soccernet/videos 
7z x test_labels.zip -o/path/to/data/soccernet/videos 
7z x valid_labels.zip -o/path/to/data/soccernet/videos 
```

---



## Using and extending the code 

If you want to extend the code to train your own model you can use following interface for accessing the data: 

```python

from dudek.data.team_bas import SoccerVideo

# load bas video 
bas_video = SoccerVideo.bas_video_from_path(
    "/path/to/data/soccernet/sn_bas_2025/england_efl/2019-2020/2019-10-01 - Brentford - Bristol City/",
    resolution=224
)

# video metadata 
print(bas_video.league) # "england_efl"
print(bas_video.season) # "2019-2020"
print(bas_video.match) # "2019-10-01 - Brentford - Bristol City"
print(bas_video.absolute_path) # "/path/to/data/soccernet/sn_bas_2025/england_efl/2019-2020/2019-10-01 - Brentford - Bristol City/224p.mp4"

# video annotations 
print(bas_video.annotations) # list of annotations

# check out single annotation
annotation  = bas_video.annotations[0]
# get frame number of that annotation  
annotation.get_frame_nr(fps=bas_video.metadata_fps)
team = annotation.team # Team.LEFT or Team.RIGHT

label = annotation.label # enum value of ActionLabel or BASLabel 


# load action spotting video (from < 2025 dataset) - this will load 2 videos as they are divided into halfs 
action_spotting_video = SoccerVideo.action_spotting_video_from_path(
    "/path/to/data/soccernet/videos/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
    resolution=224
)

# you can do pretty much the same thing with action spotting video as with bas video
# the only difference is enum class of annotation label being ActionLabel (not BASLabel)

```

You can also load all the videos from bas dataset (and optionally save frames to FS for further processing)

```python
from dudek.utils.video import load_bas_videos, load_action_spotting_videos

bas_videos = load_bas_videos(
    "/path/to/data/soccernet/sn_bas_2025/",
    resolution=224
)

# save frames to FS for further processing (this may take a while) 
for video in bas_videos:
    # by default frames are saved to FS (same dir as original video under .frames* folder) 
    video.save_frames(
        target_width=224,
        target_height=224,
        stride=2,
        grayscale=False,
    )

# use frames to generate short clips 
single_video = bas_videos[0]

# split video to continuous clips (where gap of max 2 frames is accepted) 
clips = single_video.get_clips(
    accepted_gap=2
)

# then split each clip to smaller clips of fixed size: 
fixed_length_clips = []
for clip in clips:
    fixed_length_clips += clip.split(
        clip_frames_count=200,
        overlap=190
    )

fixed_length_clip = fixed_length_clips[0]

# you can save this short video clip as video for visual inspection 
fixed_length_clip.save_as_labelled_video(
    output_path="clip.mp4", fps=12
)

# access clip frames and their attributes 
for frame in fixed_length_clip.frames:    
    print(frame.frame_path)
    print(frame.annotation)
    print(frame.original_video_frame_nr)
    # ...

```

To train any model via pytorch you may want to get inspired by baseline TDEED model: 


```python

from dudek.ml.data.tdeed import TdeedVideoClip, TeamTDeedDataset
from dudek.data.team_bas import BASLabel

# get single TDEED input clip 
tdeed_clip = TdeedVideoClip.from_video_clip(
    fixed_length_clip,
    labels_displacement=4,
    flip_proba=0.25,  # horizontal flip probability 
    camera_movement_proba=0.25,  # random camera movement probability    
    crop_proba=0.25,    
    labels_enum=BASLabel
)

# or corresponding pytorch dataset
dataset = TeamTDeedDataset(
    clips=fixed_length_clips,
    displacement=4,
    flip_proba=0.25,  # horizontal flip probability
    camera_move_proba=0.25,  # random camera movement probability    
    crop_proba=0.25,
    labels_enum=BASLabel
)

```


## How to reproduce Dudek result on SN-BAS-2025

In case u don't want to mess with the code too much you can use CLI shipped with this package to reproduce dudek results.
First thing to do is to extract frames from the videos. In case you want to pretrain on all broadcasts videos and then fine-tune on SN-BAS-2025 you need to extract frames from both datasets:

### Extracting frames

```bash
bas-frame-extract extract-bas-frames \
    --dataset_path="/path/to/data/soccernet/sn_bas_2025/" \
    --resolution=224 \
    --stride=2 \
    --frame_target_width=224 \
    --frame_target_height=224
```

```bash
bas-frame-extract extract-action-spotting-frames  \
    --dataset_path="/path/to/data/soccernet/videos/" \
    --resolution=224 \
    --stride=2 \
    --frame_target_width=224 \
    --frame_target_height=224
```

This may take a while - up to 24h depending on your hardware, please also be aware it may take a bit of your disk space (around extra 1.2TB). 

Once you are done with that, you can train a baseline model:


### Pretraining

Start by pretraining on broadcast videos:

```
bas-tdeed-train pretrain --help 
Usage: bas-tdeed-train pretrain [OPTIONS]

Options:
  --dataset_path TEXT             [required]
  --resolution INTEGER
  --clip_frames_count INTEGER
  --overlap INTEGER
  --displacement INTEGER
  --flip_proba FLOAT
  --camera_move_proba FLOAT
  --crop_proba FLOAT
  --even_choice_proba FLOAT
  --nr_epochs INTEGER
  --warm_up_epochs INTEGER
  --learning_rate FLOAT
  --train_batch_size INTEGER
  --val_batch_size INTEGER
  --eval_metric TEXT
  --start_eval_epoch_nr INTEGER
  --loss_foreground_weight INTEGER
  --features_model_name TEXT
  --temporal_shift_mode TEXT
  --acc_grad_iter INTEGER
  --enforce_train_epoch_size INTEGER
  --enforce_val_epoch_size INTEGER
  --gaussian_blur_kernel_size INTEGER
  --tdeed_arch_n_layers INTEGER
  --tdeed_arch_sgp_ks INTEGER
  --tdeed_arch_sgp_k INTEGER
  --save_every_epoch BOOLEAN
  --save_as TEXT
  --wandb_experiment_name TEXT
  --random_seed INTEGER

```

Simplest example: 

```bash
bas-tdeed-train pretrain \ 
  --dataset_path=/path/to/data/soccernet/videos/ \ 
  --resolution=224 \ 
  --save_as=pretrained.pt \ 
  --clip_frames_count=80 \ 
  --overlap=40 \ 
  --enforce_train_epoch_size=6000

```
To monitor pretraining run `tensorboard --logdir=runs` and go to http://localhost:6006/


Once you are done with that your best model will be stored as "pretrained.pt" in the current directory

### Fine-tuning
you can now fine-tune it on SN-BAS-2025:

```
bas-tdeed-train train --help                                                                                                 

Usage: bas-tdeed-train train [OPTIONS]

Options:
  --dataset_path TEXT             [required]
  --resolution INTEGER
  --clip_frames_count INTEGER
  --overlap INTEGER
  --displacement INTEGER
  --flip_proba FLOAT
  --camera_move_proba FLOAT
  --crop_proba FLOAT
  --even_choice_proba FLOAT
  --nr_epochs INTEGER
  --warm_up_epochs INTEGER
  --learning_rate FLOAT
  --train_batch_size INTEGER
  --val_batch_size INTEGER
  --eval_metric TEXT
  --start_eval_epoch_nr INTEGER
  --loss_foreground_weight INTEGER
  --features_model_name TEXT
  --temporal_shift_mode TEXT
  --acc_grad_iter INTEGER
  --enforce_train_epoch_size INTEGER
  --enforce_val_epoch_size INTEGER
  --gaussian_blur_kernel_size INTEGER
  --tdeed_arch_n_layers INTEGER
  --tdeed_arch_sgp_ks INTEGER
  --tdeed_arch_sgp_k INTEGER
  --model_checkpoint_path TEXT
  --save_as TEXT
  --experiment_name TEXT
  --random_seed INTEGER
  --help                          Show this message and exit.


```

The simplest approach is: 
```bash
 bas-tdeed-train train \ 
  --dataset_path=/path/to/data/soccernet/sn_bas_2025/ \ 
  --resolution=224 --model_checkpoint_path=pretrained.pt \ 
  --clip_frames_count=80 \
  --overlap=68 \
  --enforce_train_epoch_size=6000
```

Again, go to wandb or tensorboard web interface to monitor fine-tuning progress. Here you should see mAP for eval set instead of loss. By defaults splits are the same as those provided by challenge organizers (random_seed default param does that trick for you)

### Fine-tuning with no evaluation 

The simplest scenario to provide submission to the challenge, is to train the model on the whole dataset. In this scenario you will fine-tune the model for N epochs. 

```bash 
 bas-tdeed-train train-challenge \ 
  --dataset_path=/mnt/data4t/soccernet/sn_bas_2025/ \ 
  --resolution=224 \ 
  --model_checkpoint_path=pretrained.pt \ 
  --clip_frames_count=80 \ 
  --overlap=68 \ 
  --enforce_train_epoch_size=6000 \ 
  --nr_epochs=30 
  --save_as=tdeed_challenge.pt
 ```

### Preparing submission archive 

First, unzip challenge dataset to separate directory:

```bash 
7z x challenge.zip -o/path/to/data/soccernet/challenge_bas_data # same as before you need to provide NDA password 
```

Next, extract frames from challenge videos: 

```bash
bas-frame-extract extract-bas-frames \                                                                                                                                                                                                                                                                             kamil@matka
    --dataset_path="path/to/data/soccernet/challenge_bas_data" \
    --resolution=224 \
    --save_all=true \ 
    --stride=2 \
    --frame_target_width=224 \
    --frame_target_height=224
```

And finally we can produce submission.zip to be uploaded to codebench platform:

```bash 
bas-tdeed-train create-solution \ 
  --dataset_path=/mnt/data4t/soccernet/challenge_data \ 
  --resolution=224 \ 
  --model_checkpoint_path=tdeed_challenge.pt

```

This should produce `submission.zip` in your local directory.  Cross your fingers and upload it to codebench. 
