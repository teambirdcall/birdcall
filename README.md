# Bird Species Classification Using Bird Call Recordings

There are already many projects underway to extensively monitor birds by continuously recording natural soundscapes over long periods. However, as many living and nonliving things make noise, the analysis of these datasets is often done manually by domain experts. These analyses are painstakingly slow, and results are often incomplete. Data science may be able to assist, so researchers have turned to large crowd sourced databases of focal recordings of birds to train AI models. Unfortunately, there is a domain mismatch between the training data (short recording of individual birds) and the soundscape recordings (long recordings with often multiple species calling at the same time) used in monitoring applications. This is one of the reasons why the performance of the currently used AI models has been subpar. To unlock the full potential of these extensive and information-rich sound archives, researchers need good machine listeners to reliably extract as much information as possible to aid data-driven conservation.

# Directory Structure

 - **[audiodata](https://github.com/teambirdcall/birdcall/tree/main/audiodata)** - Contains the raw Audio files categorised by their species name as folders. This folder should contain the original audios that have not been chunked and are of variable length. 
 - **[result](https://github.com/teambirdcall/birdcall/tree/main/result)** - It contains the chunked audios as per respective species folder that have been made from audiodata folder.
 - **[melresult](https://github.com/teambirdcall/birdcall/tree/main/melresult)**- It contains the mel-spectograms of chunked audios as per respective species that have been made from result folder.
 - **[pitch_change](https://github.com/teambirdcall/birdcall/tree/main/pitch_change)** - It contains the pickle files of each respective species that have changed on the basis of pitch and dumped in this folder.
 - **[time_change](https://github.com/teambirdcall/birdcall/tree/main/time_change)** - It contains the pickle files of each respective species that have changed on the basis of time and dumped in this folder.
