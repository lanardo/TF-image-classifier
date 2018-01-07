import os
import sys
import csv
import numpy as np
import tensorflow as tf

from embed import Embed


def xaver_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def acc(d1, d2):
    cnt = 0
    for i in range(d1.__len__()):
        if d1[i] == d2[i]:
            cnt += 1
    return float(cnt)/d1.__len__()


def sel_max(data):
    ret_ind = []
    for i in range(data.__len__()):
        if data[i][0] == 1:
            ret_ind.append(0)
        else:
            ret_ind.append(1)
    return ret_ind


class Train:
    def __init__(self, train_dir="./data/train", classifier_dir="./model"):
        self.emb = Embed()

        self.train_dir = train_dir
        self.classifier_dir = classifier_dir

        self.extensions = ['.png', '.jpg']

        # self.labels = ['front', 'front_quarter', 'side', 'rear_quarter', 'rear']
        self.labels = ['front', 'front_3_quarter', 'side', 'rear_3_quarter', 'rear', 'interior', 'tire']  # only for testing

        self.rate = 0.0001
        self.epochs = 200000
        self.strip = 50

    def train_data(self):
        if not os.path.isdir(self.train_dir):
            sys.stderr.write("Not exist folder for training data\n")
            sys.exit(1)

        sub_dirs = []
        childs = os.listdir(self.train_dir)
        for child in childs:
            child_path = os.path.join(self.train_dir, child)
            if os.path.isdir(child_path):
                sub_dirs.append(child)
        sub_dirs.sort()
        self.labels = sub_dirs

        tails = []
        for i in range(len(sub_dirs)):
            line = np.zeros((len(sub_dirs)), dtype=np.uint8)
            line[i] = 1
            tails.append(line.tolist())
        """
        tails = [[1., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0.],
                 [0., 0., 1., 0., 0.],
                 [0., 0., 0., 1., 0.],
                 [0., 0., 0., 0., 1.]]
        """

        sys.stdout.write("\n Scanning folder: {}\n".format(self.train_dir))
        features = []
        for sub_dir_name in sub_dirs:
            sub_dir_path = os.path.join(self.train_dir, sub_dir_name)

            count = 0
            for fn in os.listdir(sub_dir_path):
                path = os.path.join(sub_dir_path, fn)
                if not os.path.isfile(path) or os.path.splitext(path)[1] not in self.extensions:
                    continue

                try:
                    # Extract the feature vector per each image
                    feature = self.emb.get_feature_from_image(path)
                    sys.stdout.write("\r" + path)
                    sys.stdout.flush()
                except Exception as e:
                    print(e)
                    continue
                line = feature.tolist()
                line.extend(tails[sub_dirs.index(sub_dir_name)])
                features.append(line)
                count += 1

                # if count > 10:
                #     break

            sys.stdout.write("\nLabel: {}, Counts: {}\n".format(sub_dir_name, count))

        train_data_path = os.path.join(self.train_dir, "train_data.csv")
        # save the features the csv file
        if sys.version_info[0] == 2:  # py 2x
            with open(train_data_path, 'wb') as fp:  # for python 2x
                wr = csv.writer(fp, delimiter=',')
                wr.writerows(features)
        elif sys.version_info[0] == 3:  # py 3x
            with open(train_data_path, 'w', newline='') as fp:  # for python 3x
                wr = csv.writer(fp, delimiter=',')
                wr.writerows(features)

        sys.stdout.write("Create the train_data.csv successfully!\n")
        return train_data_path, self.labels

    def train(self, bRestore):
        in_dir = self.train_dir
        out_dir = self.classifier_dir

        learning_rate = self.rate
        epochs = self.epochs
        strip = self.strip

        train_data_path = os.path.join(in_dir, 'train_data.csv')
        ckpt_path = os.path.join(out_dir, 'model_bin.ckpt')

        labels = self.labels

        x_data = []
        y_data = []

        """ Loading training data from csv files """
        print('[Step 1] Loading training data ...')
        # for python 2x
        with open(train_data_path) as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                x_data.append([float(row[i]) for i in range(0, len(row) - len(labels))])
                y_data.append([float(row[i]) for i in range(len(row) - len(labels), len(row))])

        print("total features    :" + str(len(x_data)))
        print("length of feature :" + str(len(x_data[0])))
        print("length of label   :" + str(len(y_data[0])))

        """ Placeholder """
        print('[Step 2] Placeholder')
        x = tf.placeholder('float', [None, 2048])  # len(feature) = 2048
        y = tf.placeholder('float', [None, len(labels)])  # len(Directions) = 7 : classes

        W1 = tf.get_variable('W1', shape=[2048, len(labels)], initializer=xaver_init(2048, len(labels)))
        b1 = tf.Variable(tf.zeros([len(labels)]))
        activation = tf.add(tf.matmul(x, W1), b1)
        t1 = tf.nn.softmax(activation)

        """ Minimize error using cross entropy """
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Gradient Descent

        """ Initializing the variables """
        print('[Step 3] Initializing the variables.')

        if sys.version_info[0] == 3:
            init = tf.initialize_all_variables()  # python 3x
        elif sys.version_info[0] == 2:
            init = tf.global_variables_initializer()  # python 2x

        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()

        sys.stdout.write("bRestore:{}".format(bRestore))
        if bRestore:
            print('Loading the last learning Session.')
            saver.restore(sess, ckpt_path)

        """ Training cycle """
        print('[Step 4] Training...')
        for step in range(epochs):
            sess.run(optimizer, feed_dict={x: x_data, y: y_data})
            if step % strip == 0:
                ret = sess.run(t1, feed_dict={x: x_data})
                acc1 = acc(sess.run(tf.arg_max(ret, 1)), sess.run(tf.arg_max(y_data, 1))) * 100

                sys.stdout.write("\t{} {} {}\n".format(step, sess.run(cost, feed_dict={x: x_data, y: y_data}), acc1))
                saver.save(sess, ckpt_path)
        print('Optimization Finished!')

    def save_coefs(self):
        out_dir = self.classifier_dir

        w_coef_path = os.path.join(out_dir, 'w.csv')
        b_coef_path = os.path.join(out_dir, 'b.csv')
        ckpt_path = os.path.join(out_dir, 'model_bin.ckpt')
        labels = self.labels

        """ Placeholder """
        sys.stdout.write('Placeholder\n')
        W1 = tf.get_variable('W1', shape=[2048, len(labels)], initializer=xaver_init(2048, len(labels)))
        b1 = tf.Variable(tf.zeros([len(labels)]))

        """ Initializing the variables """
        sys.stdout.write('Initializing the variables.\n')
        if sys.version_info[0] == 3:
            init = tf.global_variables_initializer()  # python 3x
        elif sys.version_info[0] == 2:
            init = tf.initialize_all_variables()  # python 2x
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()

        sys.stdout.write('Loading the last learning Session.\n')
        saver.restore(sess, ckpt_path)

        ret_W = sess.run(W1)
        ret_b = sess.run(b1)

        sys.stdout.write('Saving the coefs W, b\n')
        sys.stdout.write("{}, {}\n".format(len(ret_W), len(ret_W[0])))
        sys.stdout.write("{}\n".format(len(ret_b)))

        if sys.version_info[0] == 2:
            with open(w_coef_path, 'wb') as fp:  # for python 2x
                wr = csv.writer(fp, delimiter=',')
                wr.writerows(ret_W)
            with open(b_coef_path, 'wb') as fp:  # for python 2x
                wr = csv.writer(fp, delimiter=',')
                wr.writerow(ret_b)
        if sys.version_info[0] == 3:
            with open(w_coef_path, 'w', newline='') as fp:  # for python 3x
                wr = csv.writer(fp, delimiter=',')
                wr.writerows(ret_W)
            with open(b_coef_path, 'w', newline='') as fp:  # for python 3x
                wr = csv.writer(fp, delimiter=',')
                wr.writerow(ret_b)

        sys.stdout.write('Saving coeficient files Finished!\n')


if __name__ == '__main__':

    tr = Train()
    # tr.train_data()
    # tr.train(bRestore=True)
    tr.save_coefs()
