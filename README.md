# Photo Stylist
Stylizing yours photos using neural style transfer.

## Huge thanks to The Lazy Programmer without whose Udemy course this project would have not been possible.

## Outcome
Watch it <a href='https://youtu.be/pisXtDwLVoU'>here</a>.

## Requirements
0. Python 3.x
1. <a href="https://tensorflow.org">Tensorflow 1.5</a>
2. <a href="https://keras.io">Keras</a>
3. OpenCV 3.4
4. A good grasp over the above 5 topics along with neural networks. Refer to the internet if you have problems with those. I myself am just a begineer in those.
5. A good CPU (preferably with a GPU).
6. Patience.... A lot of it.

## How to use
Using this repo is very easy. First put your content and style images in the content and syle folder respectively. Then all you have to do is run the photo_stylist.py file.

    python photo_stylist.py

You will be asked to select the content image from the content folder and the style image from the style folder. After selecting them you are good to go.

In case you are a developer and you are not satisfied with the resultant image open up the photo_stylist.py and tweak the tweakable parameters.

## Limitations
I don't know if this exactly qualifies as a limitation but I am stil going to put it here. Since neural style tranfer is a resource intensive task, having a good CPU with a good GPU is highly recommended. In my Asus A541UJ which has a NVidia 920M graphics card, 4 GB RAM and Intel Core i3 2GHz processor any content image bigger than 500x500 px crashed the program due to non memory allocation. I don't know about any other specifications though.