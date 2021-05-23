
import argparse
import train
import image_augmentation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do", default='train', type = str, help="train, test, image_augmentation")
    parser.add_argument("--input_shape", default=[256,256,3], type=list)
    parser.add_argument("--label_shape", default=[256,256,1], type=list)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epoch", default=500, type=int)
    parser.add_argument("--aug_size", default=30, type=int)

    args = parser.parse_args()

    if args.do == "train" :
        train.train(args)
    if args.do == "test" :
        train.test(args)
    if args.do == "image_augmentation" :
        image_augmentation.image_augmentation(args)

if __name__ == "__main__":
    main()
