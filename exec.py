#! -*- coding: utf-8 -*-
from tensorflow.python.platform import flags

from tensorflow.python.ops.control_flow_util_v2 import output_all_intermediates
from tensorflow.python.platform.app import run
from model import *
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping

import numpy as np
import time
import json
import os

output_all_intermediates()

flags.DEFINE_boolean("train", False, "train or predict")
flags.DEFINE_string("train_set", "dataset/rec_train_set.sample.txt", "the file path of train set")
flags.DEFINE_string("validation_set", "dataset/rec_validation_set.sample.txt", "the file path of validation set")
flags.DEFINE_string("test_set", "dataset/test.sample.txt", "the file path of test set")
flags.DEFINE_string("saved_model_name", "drr_model.h5", "the saved model name")
flags.DEFINE_integer("batch_size", 512, "batch size for training")
flags.DEFINE_integer("seq_len", 30, "the length of input list")
flags.DEFINE_integer("train_epochs", 100, "epoch for training")
flags.DEFINE_integer("train_steps_per_epoch", 1000, "steps per epoch for training")
flags.DEFINE_integer("validation_steps", 2000, "steps for validation")
flags.DEFINE_integer("early_stop_patience", 10, "early stop when model is not improved with X epochs")
flags.DEFINE_integer("lr_per_step", 4000, "update learning rate per X step")

flags.DEFINE_integer("d_feature", 12, "the feature length of each item in the input list")
flags.DEFINE_integer("d_model", 64, "param used drr_model")
flags.DEFINE_integer("d_inner_hid", 128, "param used in drr_model")
flags.DEFINE_integer("n_head", 1, "param used in drr_model")
flags.DEFINE_integer("d_k", 64, "param used in drr_model")
flags.DEFINE_integer("d_v", 64, "param used in drr_model")
flags.DEFINE_integer("n_layers", 4, "param used in drr_model")
flags.DEFINE_float("dropout", 0.1, "param used in drr_model")
flags.DEFINE_integer("pos_embedding_mode", 1, "param use d in drr_model")  # 0:no PV  1:use PV

FLAGS = flags.FLAGS

FEATURE_INFO_MAP = {
    "icf": ['icf1', 'icf2', 'icf3', 'icf4', 'icf5'],
    "ucf": ['ucf1', 'ufc2', 'ucf3'],
    "iv": ['iv1', 'iv2', 'iv3', 'iv4', 'iv5', 'iv6', 'iv7', 'iv8', 'iv9', 'iv10', 'iv11', 'iv12'],
    "pv": ['pv1', 'pv2', 'pv3', 'pv4', 'pv5', 'pv6', 'pv7'],
    "iv+pv": ['iv1', 'iv2', 'iv3', 'iv4', 'iv5', 'iv6', 'iv7', 'iv8', 'iv9', 'iv10', 'iv11', 'iv12', 'pv1', 'pv2',
              'pv3', 'pv4', 'pv5', 'pv6', 'pv7']
}


def get_pos(batch_size, seq_len):
    outputs = np.zeros((batch_size, seq_len), dtype=np.int32)
    for i in range(batch_size):
        outputs[i] = np.arange(seq_len, dtype=np.int32)
    return outputs


def get_label(label_batch, batch_size, seq_len):
    outputs = np.zeros((batch_size, seq_len))
    i = 0
    for row in label_batch:
        outputs[i] = np.array(json.loads(row.numpy()))
        i += 1
    return outputs


def get_uid(features_batch, batch_size, seq_len):
    outputs = np.zeros((batch_size, seq_len), dtype=np.int32)
    i = 0
    for uid in features_batch:
        outputs[i] = np.array([uid] * seq_len, dtype=np.int32)
        i += 1
    return outputs


def get_icf(features_batch, batch_size, seq_len):
    global FEATURE_INFO_MAP
    feature_len = len(FEATURE_INFO_MAP["icf"])
    outputs = []
    for i in range(feature_len):
        outputs.append(np.zeros((batch_size, seq_len), dtype=np.int32))
    j = 0
    for row in features_batch:
        feature_data = np.array(json.loads(row.numpy()), dtype=np.int32).T
        for i in range(feature_len):
            outputs[i][j] = feature_data[i, :]
        j += 1
    return outputs


def get_ucf(features_batch, batch_size, seq_len):
    global FEATURE_INFO_MAP
    feature_len = len(FEATURE_INFO_MAP['ucf'])
    outputs = []
    for i in range(feature_len):
        outputs.append(np.zeros((batch_size, seq_len), dtype=np.int32))
    j = 0
    for row in features_batch:
        feature_data = np.tile(
            np.array(json.loads(row.numpy().replace(bytes("null", encoding='utf8'), bytes('0', encoding='utf8'))),
                     dtype=np.int32), (seq_len, 1)).T
        for i in range(feature_len):
            outputs[i][j] = feature_data[i, :]
        j += 1
    return outputs


def get_iv(features_batch, batch_size, seq_len):
    global FEATURE_INFO_MAP
    feature_len = len(FEATURE_INFO_MAP['iv'])
    outputs = np.zeros((batch_size, seq_len, feature_len))
    i = 0
    for row in features_batch:
        outputs[i] = np.array(json.loads(row.numpy()))
        i += 1
    return outputs


def get_pv(features_batch, batch_size, seq_len):
    global FEATURE_INFO_MAP
    feature_len = len(FEATURE_INFO_MAP['pv'])
    outputs = np.zeros((batch_size, seq_len, feature_len))
    i = 0
    for row in features_batch:
        outputs[i] = np.array(json.loads(row.numpy()))
        i += 1
    return outputs


def get_iv_and_pv(iv_batch, pv_batch, batch_size, seq_len):
    iv = get_iv(iv_batch, batch_size, seq_len)
    pv = get_pv(pv_batch, batch_size, seq_len)
    return np.dstack((iv, pv))


def get_features(uid_batch, ucf_batch, icf_batch, pv_batch, iv_batch, batch_size, seq_len):
    if FLAGS.pos_embedding_mode == 0:
        outputs = [get_pos(batch_size, seq_len), get_iv(iv_batch, batch_size, seq_len)]
        assert FLAGS.d_feature == len(FEATURE_INFO_MAP['iv'])
        return outputs
    else:
        outputs = [get_pos(batch_size, seq_len), get_iv_and_pv(iv_batch, pv_batch, batch_size, seq_len)]
        assert FLAGS.d_feature == len(FEATURE_INFO_MAP['iv']) + len(FEATURE_INFO_MAP['pv'])
        return outputs


def input_generator(filename, batch_size, seq_len, repeat_cnt=-1):
    print("data_set={0} batch_size={1} seq_len={2} repeat_cnt={3} for input_generator".format(filename, batch_size,
                                                                                              seq_len, repeat_cnt))

    dataset = tf.data.experimental.CsvDataset([filename], record_defaults=[0, "", "", "", "", ""], field_delim='|') \
        .repeat(repeat_cnt).batch(batch_size)

    iterator = iter(dataset)
    while True:
        try:
            uid_batch, ucf_batch, icf_batch, pv_batch, iv_batch, label_batch = next(iterator)
            yield get_features(uid_batch, ucf_batch, icf_batch, pv_batch, iv_batch, batch_size, seq_len), get_label(
                label_batch, batch_size, seq_len)
        except StopIteration:
            iterator = iter(dataset)


def get_model():
    model = None
    _model = PRM(FLAGS.seq_len, FLAGS.d_feature, d_model=FLAGS.d_model, d_inner_hid=FLAGS.d_inner_hid,
                 n_head=FLAGS.n_head, d_k=FLAGS.d_k, d_v=FLAGS.d_v, layers=FLAGS.n_layers, dropout=FLAGS.dropout)
    model = _model.build(pos_mode=FLAGS.pos_embedding_mode)
    model.summary()
    print(
        "model_setting:\n\tseq_len={0}\n\td_feature={1}\n\td_model={2}\n\td_inner_hid={3}\n\tn_head={4}\n\td_k={5}"
        "\n\td_v={6}\n\tn_layers={7}\n\tdropout={8}\n\tpos_embedding_mode={9}".format(
            FLAGS.seq_len, FLAGS.d_feature, FLAGS.d_model, FLAGS.d_inner_hid, FLAGS.n_head, FLAGS.d_k, FLAGS.d_v,
            FLAGS.n_layers, FLAGS.dropout, FLAGS.pos_embedding_mode))
    print("-" * 98)
    return model


class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        super().__init__()
        self.basic = d_model ** -0.5
        self.warm = warmup ** -1.5
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


class LRSchedulerPerEpoch(Callback):
    def __init__(self, d_model, warmup=4000, num_per_epoch=1000):
        super().__init__()
        self.basic = d_model ** -0.5
        self.warm = warmup ** -1.5
        self.num_per_epoch = num_per_epoch
        self.step_num = 1

    def on_epoch_begin(self, epoch, logs=None):
        self.step_num += self.num_per_epoch
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


def train():
    print("training....")
    if not os.path.exists(r".\logs"):
        os.mkdir(r".\logs")
        print("create log directory:{0}".format(r".\logs"))
    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    assert FLAGS.train_set != ""
    assert FLAGS.validation_set != ""
    print("train_set={0} validation_set={1} batch_size={2} seq_len={3}".format(FLAGS.train_set,
                                                                               FLAGS.validation_set, FLAGS.batch_size,
                                                                               FLAGS.seq_len))
    train_gen = input_generator(FLAGS.train_set, FLAGS.batch_size, FLAGS.seq_len)
    next(train_gen)
    validation_gen = input_generator(FLAGS.validation_set, FLAGS.batch_size, FLAGS.seq_len)
    next(validation_gen)
    print("saved_model_name={0} early_stop_patience={1} lr_per_step={2}".format(FLAGS.saved_model_name,
                                                                                FLAGS.early_stop_patience,
                                                                                FLAGS.lr_per_step))
    callback_list = [TensorBoard(log_dir=r".\logs"),
                     ModelCheckpoint(FLAGS.saved_model_name, verbose=1, monitor='val_loss', save_weights_only=True,
                                     save_best_only=True),
                     EarlyStopping(monitor='val_loss', patience=FLAGS.early_stop_patience, verbose=1),
                     LRSchedulerPerStep(FLAGS.d_model, FLAGS.lr_per_step)]
    print("train_epochs={0} train_steps_per_epoch={1} validation_steps={2}".format(FLAGS.train_epochs,
                                                                                   FLAGS.train_steps_per_epoch,
                                                                                   FLAGS.validation_steps))

    model.fit(train_gen, epochs=FLAGS.train_epochs, steps_per_epoch=FLAGS.train_steps_per_epoch,
              verbose=1, callbacks=callback_list, validation_data=validation_gen,
              validation_steps=FLAGS.validation_steps)
    K.clear_session()
    print("finish training!")


def predict():
    print("predicting...")
    if not os.path.exists(FLAGS.saved_model_name):
        print("the model file {0} does not exist!".format(FLAGS.saved_model_name))
        return
    else:
        print("load model from {0}!".format(FLAGS.saved_model_name))
    model = get_model()
    model.load_weights(FLAGS.saved_model_name)
    assert FLAGS.test_set != ""
    test_gen = input_generator(FLAGS.test_set, FLAGS.batch_size, FLAGS.seq_len, 1)
    batch_cnt = 0
    predict_output_file = "%s.predict.out" % FLAGS.test_set
    fout = open(predict_output_file, "w")
    try:
        for test_batch in test_gen:
            batch_cnt += 1
            features_batch = test_batch[0]
            label_batch = test_batch[1]
            predict_batch = model.predict_on_batch(features_batch)
            print("processed {0} batches...".format(batch_cnt))
            for labels, predicts in zip(label_batch, predict_batch):
                if sum(labels) > 0:  # predict valid labels 
                    new_ranks = np.argsort(-predicts)
                    new_labels = labels[new_ranks]
                    fout.write("%s\t%s\n" % (json.dumps(labels.tolist()), json.dumps(new_labels.tolist())))
    except tf.errors.OutOfRangeError:
        print("finish predicting!")
    fout.close()
    return 0


def main(_):
    beg_time = time.time()
    if FLAGS.train:
        train()
    else:
        predict()
        # get_model()
    time_cost = (time.time() - beg_time) / 60
    print("job done! time_cost={0} minutes".format(round(time_cost)))


if __name__ == "__main__":
    run()
