# PySignTranslatEr
it's intended to be used for sign language translation, but it's undertrained

### at the time of making this document translator "knew" only 8 words(no, why not, yes, taxi, no thanks, night, niger(country), good day) and unknown word/not a word mark(not sign)

# this is only a proof of concept


## to run it you need

an archive with weights, which can be found here [link to weight archive(it was too big for github)](https://drive.google.com/file/d/1f8WeMSNqRkSZRjwMI1PCs1KHATk_jGwn/view?usp=sharing), extract this archive to the directory with all other scripts

It requires:
* Python3
* Tensorflow
* Tflearn
* PIL
* Opencv2
* Matplotlib
* Numpy

## to use it

**UPDATE**: almost everything was automated and added to **translate.py**

to use it on a video file, just run **translate.py** with ```-vp``` argument and a path to your video file
```
python3 translate.py -vp path/to/your/video/file.mp4
```
 
the result will be saved in cronological order to **tranlsated.txt**, raw output from the neural network will be saved in cronological order to **translated_raw.txt**

Video formats supported(tested):
* ```mp4```
* ```mkv```
* ```avi```

## how it works

it uses 2 neural networks, 1 for cutting the video into words and 1 for translating words.
both neural networks are lstm's with one hiden layer

their design is not otimal and both are undertrained, but it works as a proof of concept

## to train it

trainning is split into two, *cutter* trainning and *translator* trainning

#### to train the *cutter*(NOT RECOMMENDED)
you need to put your data, with labels in .log files that have the same name as your video files, in **temp2_train** folder in the directory with all scripts from this project

to generate .log files you would have to use **marker.py** BUT it would need cunfiguration and some video editing on your side

i will make a detailed doc on how to do that one day

after you oranised the trainning data run **CutNN_train.py** to train it
```
python3 CutNN_train.py
```


#### to train the *translator*
put your data, with labels in the file name, in **temp_train** folder in the directory with all scripts from this project

there is no way to add more words atm(again design of this model is not optimal. output layer is curently configured as 1 neuron per label)

***DON'T MIX LABELS***

List of labels:
* ```no``` - no
* ```whynz``` - why not
* ```yes``` - yes
* ```taxi``` - taxi
* ```nthx``` - no thanks
* ```night``` - night
* ```niger``` - niger
* ```goodday``` - good day
* ```nx sign``` - not sign

run **lstm9_train.py** to train it
```
python3 lstm9_train.py
```


### microsoft sign language translator
[link to the paper](https://pdfs.semanticscholar.org/165a/1a7c529f51b91ae587496d41603e560d1fe9.pdf)
