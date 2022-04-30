# Image_Caption_Wiki

## Project Organization

```
root directory
	├── model.py               # construct Encoder(ResNet101), Attention, Decoder(LSTM)
	├── Main.py                # main function to train a model
	├── utils.py               # helper functions
	├──Wiki_Dataset.py	   # customized dataset
	└── inferrence.py          # predict with one image
```

### Pre_trained weight Link
https://drive.google.com/file/d/1-eI3szeX-5__2B7p_Upq_ObmgPJmBIU6/view?usp=sharing

### word_map link
https://drive.google.com/file/d/1oh_hs1gSZdEBugpT3qkyqeXvePo92jAX/view?usp=sharing

## How to use
## Train:
### Step one: pip relative packages in Main.py
### Step two: keep word map, pre_trained weight and Main.py in the same root

## Inferrence:
### Keep pre_trained weight, word_map in inferrence.py in the same root, set image_path as the path of the image you want to predict.

## Colab path
