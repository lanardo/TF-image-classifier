import tensorflow as tf
import os
import numpy as np
import sys
import tarfile
import cv2
import csv
from six.moves import urllib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for disable the info log of tensorflow itself
MODEL_DIR = "./model"

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
LABELS = ['front', 'front_quarter', 'side', 'rear_quarter', 'rear']
directions = [
    [1., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 1.]
]


def create_graph():
    # Creates a graph from saved GraphDef file and returns a saver.
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def maybe_download_and_extract():
    # Download and extract model tar file.
    dest_directory = MODEL_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('    \r>> Downloading %s %.1f%%\n' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)

        statinfo = os.stat(filepath)
        sys.stdout.write('    Successfully downloaded' + filename + statinfo.st_size + 'bytes.\n')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


sess, softmax_tensor = None, None


def _get_feature_from_image(img_path):
    # Runs extract the feature from the image.
    # Args: img_path: Image file name.
    # Returns:  predictions: 2048 * 1 feature vector
    global sess, softmax_tensor

    if not tf.gfile.Exists(img_path):
        tf.logging.fatal('File does not exist %s', img_path)
    image_data = tf.gfile.FastGFile(img_path, 'rb').read()

    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    return predictions


w, b = None, None


def _get_label(feature):
    global w, b
    ret = []
    # for python 2x
    for i in range(len(b[0])):
        ele = 0.0
        for j in range(len(w)):
            ele += feature[j]*w[j][i]
        ret.append(ele + b[0][i])
    return ret.index(max(ret))


def _load_model_coefs(model_dir):
    global w, b
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
    return True

    
def _inference(input, output):
    tf.reset_default_graph()
    tf.Graph().as_default()

    global sess, softmax_tensor

    maybe_download_and_extract()

    create_graph()
    sess = tf.Session()
    softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
     
    if not os.path.exists(output):
        os.mkdirs(output)
    
    # scan image in input
    cnt = [0, 0, 0, 0, 0]
    sys.stdout.write("    Scan folder for classification...\n")
    
    files = [f for f in os.listdir(input) if os.path.isfile(os.path.join(input, f))]
    sys.stdout.write("    total files {}\n".format(len(files)))

    for f in files:
        fn, ext = os.path.splitext(f)
        if ext.lower() == '.png':
            # Convert image file to jpg
            im = cv2.imread(os.path.join(input, f))
            new_fn = os.path.join(input, fn + '.jpg')
            cv2.imwrite(new_fn, im)
            os.remove(os.path.join(input, f))
        elif ext.lower() == '.jpg':
            new_fn = os.path.join(input, f)
        else:
            continue

        # Extract the feature vector per each image
        feature = _get_feature_from_image(new_fn)
        line = feature.tolist()
        label_idx = _get_label(line)
        
        # move the labeled image to folder named with its label(string)
        cnt[label_idx] += 1
        label = LABELS[label_idx]

        label_dir = os.path.join(output, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        out_fn = os.path.join(label_dir, f)

        if os.path.exists(out_fn):
            os.remove(new_fn)
        else:
            os.rename(new_fn, out_fn)
        sys.stdout.write("    label({}) : file{}\n".format(label, new_fn))

    return cnt


def classify(input_dir, output_dir, model_dir):

    """
       classify the input image to 7 classes
           LABELS = ['front', 'front_3_quarter', 'side', 'rear_3_quarter', 'rear', 'interior', 'tire']
       Args:
           input_dir:
           output_dir:

       Returns:
           list:
            result of classification number of images for each labels

    """

    if input_dir is None or output_dir is None:
        raise Exception("    Input_dir or Output_dir not defined")

    if model_dir is None:
        model_dir = "./mdoel"
    global MODEL_DIR
    MODEL_DIR = model_dir
    if not os.path.exists(model_dir) or _load_model_coefs(model_dir) == False:
        raise Exception("    No such dir or files for getting coefs.")

    cnts = _inference(input_dir, output_dir)
    sys.stdout.write("    Classification finished Successfully!\n")

    return cnts
