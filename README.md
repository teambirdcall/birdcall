# Bird Species Classification Using Bird Call Recordings
There are already many projects underway to extensively monitor birds by continuously recording natural soundscapes over long periods. However, as many living and nonliving things make noise, the analysis of these datasets is often done manually by domain experts. These analyses are painstakingly slow, and results are often incomplete. Data science may be able to assist, so researchers have turned to large crowd sourced databases of focal recordings of birds to train AI models. Unfortunately, there is a domain mismatch between the training data (short recording of individual birds) and the soundscape recordings (long recordings with often multiple species calling at the same time) used in monitoring applications. This is one of the reasons why the performance of the currently used AI models has been subpar. To unlock the full potential of these extensive and information-rich sound archives, researchers need good machine listeners to reliably extract as much information as possible to aid data-driven conservation.

# Directory Structure

-  **[audiodata](https://github.com/teambirdcall/birdcall/tree/main/audiodata)** - Contains the raw Audio files categorized by their species name as folders. This folder should contain the original audios that have not been chunked and are of variable length.

-  **[result](https://github.com/teambirdcall/birdcall/tree/main/result)** - It contains the chunked audios as per respective species folder that have been made from audiodata folder.

-  **[melresult](https://github.com/teambirdcall/birdcall/tree/main/melresult)**- It contains the mel-spectrogram of chunked audios as per respective species that have been made from result folder.

-  **[pitch_change](https://github.com/teambirdcall/birdcall/tree/main/pitch_change)** - It contains the pickle files of each respective species that have changed on the basis of pitch and dumped in this folder.

-  **[time_change](https://github.com/teambirdcall/birdcall/tree/main/time_change)** - It contains the pickle files of each respective species that have changed on the basis of time and dumped in this folder.

-  **[mfcc](https://github.com/teambirdcall/birdcall/tree/main/mfcc)** - It contains the mfcc-data in the form of pickle files of all the species in the result folder.

## Steps to follow :scroll:

### 1. Fork it :fork_and_knife:
You can get your own fork/copy of [BirdCall](https://github.com/teambirdcall/birdcall) by using the <kbd><b>Fork</b></kbd> button 
 [![Fork Button](https://help.github.com/assets/images/help/repository/fork_button.jpg)](https://github.com/teambirdcall/birdcall)
 
 
### 2. Clone it :busts_in_silhouette:
You need to clone (download) it to local machine using 
```sh
$ git clone https://github.com/Your_Username/birdcall.git
```
> This makes a local copy of repository in your machine.
Once you have cloned the `birdCall` repository in Github, move to that folder first using change directory command on linux and Mac.
```sh
# This will change directory to a folder birdcall
$ cd birdcall
```
Move to this folder for all other commands.

### 3. Set it up :arrow_up:
Run the following commands to see that *your local copy* has a reference to *your forked remote repository* in Github :octocat:
```sh
$ git remote -v
origin  https://github.com/Your_Username/birdcall.git (fetch)
origin  https://github.com/Your_Username/birdcall.git (push)
```
Now, lets add a reference to the original [birdcall](https://github.com/teambirdcall/birdcall) repository using
```sh
$ git remote add upstream https://github.com/teambirdcall/birdcall.git
```
> This adds a new remote named ***upstream***.
See the changes using
```sh
$ git remote -v
origin    https://github.com/Your_Username/birdcall.git (fetch)
origin    https://github.com/Your_Username/birdcall.git (push)
upstream  https://github.com/teambirdcall/birdcall.git (fetch)
upstream  https://github.com/teambirdcall/birdcall.git (push)
```

### 4. Ready Steady Go... :turtle: :rabbit2:
Once you have completed these steps, you are ready to start contributing to our repository by creating [pull requests](https://github.com/teambirdcall/birdcall/pulls).

### 5. Create a new branch :bangbang:
Whenever you are going to make contribution. Please create separate branch using command and keep your `master` branch clean (i.e. synced with remote branch).
```sh
# It will create a new branch with name Branch_Name and switch to branch Folder_Name
$ git checkout -b Folder_Name
```
Create a separate branch for contribution and try to use same name of branch as of folder.
To switch to desired branch
```sh
# To switch from one folder to other
$ git checkout Folder_Name
```
To add the changes to the branch. Use
```sh
# To add all files to branch Folder_Name
$ git add .
```
Type in a message relevant for the code reviewer using
```sh
# This message get associated with all files you have changed
$ git commit -m 'relevant message'
```
Now, Push your awesome work to your remote repository using
```sh
# To push your work to your remote repository
$ git push -u origin Folder_Name
```
Finally, go to your repository in browser and click on `compare and pull requests`.
Then add a title and description to your pull request that explains your precious effort.
