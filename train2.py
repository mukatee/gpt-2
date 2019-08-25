#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2

import model, sample, encoder
from load_dataset import load_dataset, Sampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context

def generate_samples(data_sampler, generate_from, args, sess, tf_sample, context, enc, counter):
    print('Generating samples...')
    context_tokens = data_sampler.sample(1)
    if generate_from is not None:
        context_tokens = enc.encode(generate_from)
    all_text = []
    index = 0
    while index < args.sample_num:
        out = sess.run(
            tf_sample,
            feed_dict={context: args.batch_size * [context_tokens]})
        for i in range(min(args.sample_num - index, args.batch_size)):
            text = enc.decode(out[i])
            text = '======== SAMPLE {} ========\n{}\n'.format(
                index + 1, text)
            all_text.append(text)
            index += 1
    print(text)
    maketree(os.path.join(SAMPLE_DIR, args.run_name))
    with open(
            os.path.join(SAMPLE_DIR, args.run_name,
                         'samples-{}').format(counter), 'w') as fp:
        fp.write('\n'.join(all_text))

class Object(object):
    pass

args = Object()
args.model_name = "117M"
args.restore_from = "../talkingdonkeys/gpt2-models/lyrics"
args.optimizer = "adam"
args.batch_size = 1
args.noise = 0.0
args.sample_length = 200
args.top_k = 40
args.top_p = 0.0
args.only_train_transformer_layers = False
args.learning_rate = 0.00002
args.accumulate_gradients = 1
args.memory_saving_gradients = False
args.dataset = "texts.npz"
args.combine = 50000
args.run_name = "run1"
args.sample_num = 1

enc = encoder.get_encoder(args.model_name)
hparams = model.default_hparams()
generate_from = "hello world"
with open(os.path.join('models', args.model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if args.model_name == '345M':
    args.memory_saving_gradients = True
    if args.optimizer == 'adam':
        args.only_train_transformer_layers = True

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF

sess = tf.Session(config=config)

def bootstrap(sess, args, hparams, enc, generate_from):

    context = tf.placeholder(tf.int32, [args.batch_size, None])
    context_in = randomize(context, hparams, args.noise)
    output = model.model(hparams=hparams, X=context_in)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    tf_sample = sample.sample_sequence(
        hparams=hparams,
        length=args.sample_length,
        context=context,
        batch_size=args.batch_size,
        temperature=1.0,
        top_k=args.top_k,
        top_p=args.top_p)

    all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
    train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars

    if args.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    elif args.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
    else:
        exit('Bad optimizer:', args.optimizer)

    if args.accumulate_gradients > 1:
        if args.memory_saving_gradients:
            exit("Memory saving gradients are not implemented for gradient accumulation yet.")
        opt = AccumulatingOptimizer(
            opt=opt,
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients()
        summary_loss = tf.summary.scalar('loss', opt_apply)
    else:
        if args.memory_saving_gradients:
            opt_grads = memory_saving_gradients.gradients(loss, train_vars)
        else:
            opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.summary.scalar('loss', loss)

    summary_lr = tf.summary.scalar('learning_rate', args.learning_rate)
    summaries = tf.summary.merge([summary_lr, summary_loss])

    saver = tf.train.Saver(
        var_list=all_vars,
        max_to_keep=5,
        keep_checkpoint_every_n_hours=2)
    sess.run(tf.global_variables_initializer())


    ckpt = tf.train.latest_checkpoint(args.restore_from)
    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)

    print('Loading dataset...')
    chunks = load_dataset(enc, args.dataset, args.combine)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    print('Training...')

    counter = 1
    counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
    if os.path.exists(counter_path):
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1

    def generate_samples():
        print('Generating samples...')
        context_tokens = data_sampler.sample(1)
        if generate_from is not None:
            context_tokens = enc.encode(generate_from)
        all_text = []
        index = 0
        while index < args.sample_num:
            out = sess.run(
                tf_sample,
                feed_dict={context: args.batch_size * [context_tokens]})
            for i in range(min(args.sample_num - index, args.batch_size)):
                text = enc.decode(out[i])
                text = '======== SAMPLE {} ========\n{}\n'.format(
                    index + 1, text)
                all_text.append(text)
                index += 1
        print(text)
        maketree(os.path.join(SAMPLE_DIR, args.run_name))
        with open(
                os.path.join(SAMPLE_DIR, args.run_name,
                             'samples-{}').format(counter), 'w') as fp:
            fp.write('\n'.join(all_text))

    def sample_batch():
        return [data_sampler.sample(1024) for _ in range(args.batch_size)]


    while True:
        if counter > 1:
            generate_samples()
            return data_sampler, generate_from, args, sess, tf_sample, context, enc, counter

        if args.accumulate_gradients > 1:
            sess.run(opt_reset)
            for _ in range(args.accumulate_gradients):
                sess.run(
                    opt_compute, feed_dict={context: sample_batch()})
            (v_loss, v_summary) = sess.run((opt_apply, summaries))
        else:
            (_, v_loss, v_summary) = sess.run(
                (opt_apply, loss, summaries),
                feed_dict={context: sample_batch()})

        counter += 1

data_sampler, generate_from, args, sess, tf_sample, context, enc, counter = bootstrap(sess, args, hparams, enc, generate_from)

def generate_sample():
    return generate_samples(data_sampler, generate_from, args, sess, tf_sample, context, enc, counter)

if __name__ == '__main__':
    print()
    print("sample1 from main:")
    generate_sample()
    print()
    print("sample2 from main:")
    generate_sample()
