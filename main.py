from util_train import *
from util_normal import *
import feature_extraction

# import json

def main():
    device = randomseed()

    # exit()

    print('1: temp test, 2: real train, 3: train continue, 4: feature extraction, 5: Batch size tester, else: exit')
    x=int(input())
    global hyparam

    hyparam=open_yaml('./hyparam.yaml')
    hyparam['device']=device


    if x==1:
        temp_train_prepare(hyparam)
    elif x==2:
        train_prepare(hyparam)
    elif x==3:
        _ = randomseed(seed=7)
        train_prepare(hyparam, train_continue=True)
    elif x==4:
        feature_extraction.extraction(hyparam)
    elif x == 5:
        x=batch_size_tester(hyparam, max_size=256)
        x.forward()
    else:
        print('exit!!')


if __name__=="__main__":

    main()