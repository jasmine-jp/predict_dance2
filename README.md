## What is predict_dance ?
***predict_dance*** is enable to decide which videoes belong to ***[elegant, dance, other]***.
## How do I start ?
first of all, you need to install **dependencies**.<br>
you should type this command.
```
pip install -r src/git/requirements.txt
```
and you should install **ffmpeg**, **librosa**.
```
brew install ffmpeg
```
```
brew install libsndfile
```
next, you need to create directories.<br>
plz check tree down below.
<pre>
.
├── archive
│   ├── *.mp4 (for stock)
├── out
│   ├── edited
│   │   ├── sound
│   │   │   ├── *.mp3
│   │   ├── video
│   │   │   ├── *.mp4
│   ├── img (this directory is created automatically)
│   │   ├── epoch_n
│   │   │   ├── *.png
│   ├── model
│   │   ├── *.pkl
│   └── src
│   │   ├── sound
│   │   │   ├── *.pkl
│   │   ├── video
│   │   │   ├── *.pkl
├── test
│   ├── *.mp4 (for test)
└── video
    ├── *.mp4 (for prediction)
</pre>
Finally, you checkout branches.<br>
if you type these command, you can execute prediction.
```
python main.py
```
```
python test.py
```

you can modify code everything. plz read my code and improve better!<br>
I show you recent nn for help.
## flow chart
![flowchart](src/git/flowchart.png)