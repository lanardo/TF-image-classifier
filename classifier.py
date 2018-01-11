import os
import sys
import cv2
import csv

from embed import Embed


def classify(feature, w, b):
    ret = []
    # for python 2x
    for i in range(len(b[0])):
        ele = 0.0
        for j in range(len(w)):
            ele += feature[j]*w[j][i]
        ret.append(ele + b[0][i])
    return ret.index(max(ret))


def load_coefs(model_dir):

    w_fn = os.path.join(model_dir, "w.csv")
    b_fn = os.path.join(model_dir, "b.csv")
    sys.stdout.write("    file for W : {}\n".format(w_fn))
    sys.stdout.write("    file for b :{}.\n".format(b_fn))

    if not os.path.isfile(w_fn) or not os.path.isfile(b_fn):
        return False

    w = []
    with open(w_fn) as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        for row in csv_reader:
            w.append([float(row[i]) for i in range(len(row))])
    sys.stdout.write("    shape of W : {} x {}\n".format(len(w), len(w[0])))
    b = []
    with open(b_fn) as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        for row in csv_reader:
            b.append([float(row[i]) for i in range(len(row))])
    sys.stdout.write("    shape of b : {} x {}\n".format(len(b), len(b[0])))
    return w, b


def load_label_info(train_label_path):
    labels = []
    with open(train_label_path, 'r') as fp:
        for line in fp:
            line = line.replace('\n', '')
            labels.append(line)
        return labels


def inference(in_dir, model_dir):
    emb = Embed()

    LABELS = load_label_info("./data/train/train_label.txt")
    w, b = load_coefs(model_dir)

    cnts = [0, 0, 0, 0, 0]
    # scan image in input
    sys.stdout.write("    Scan folder for classification...\n")
    files = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    sys.stdout.write("    total files {}\n".format(len(files)))

    for f in files:
        fn, ext = os.path.splitext(f)
        if ext.lower() == '.png':
            # Convert image file to jpg
            im = cv2.imread(os.path.join(in_dir, f))
            new_fn = os.path.join(in_dir, fn + '.jpg')
            cv2.imwrite(new_fn, im)
            os.remove(os.path.join(in_dir, f))
        elif ext.lower() == '.jpg':
            new_fn = os.path.join(in_dir, f)
        else:
            continue

        # Extract the feature vector per each image
        feature = emb.get_feature_from_image(new_fn)
        feature = feature.tolist()
        label_idx = classify(feature, w, b)
        
        # move the labeled image to folder named with its label(string)
        cnts[label_idx] += 1
        label = LABELS[label_idx]

        label_dir = os.path.join(in_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        out_fn = os.path.join(label_dir, f)

        if os.path.exists(out_fn):
            os.remove(new_fn)
        else:
            os.rename(new_fn, out_fn)
        sys.stdout.write("    label({}) : file{}\n".format(label, new_fn))

    return cnts


if __name__ == '__main__':
    inference("./data/test", "./model")