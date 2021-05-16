import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.
This is an implementation of the method described in
Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.
Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """
        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================
            Define the computational graph with necessary variables.

            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()

            self.train_inputs =
            self.train_labels =
            self.test_inputs =
            ...


            2) Call forward_pass and get predictions

            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)
            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam

            ...
            self.loss =

            ===================================================================
            """

            batch_size, n_Tokens, n_Transitions = Config.batch_size, Config.n_Tokens, parsing_system.numTransitions()
            embedding_size, hidden_size = Config.embedding_size, Config.hidden_size

            self.train_inputs = tf.placeholder(tf.int32,shape=(batch_size,n_Tokens))
            self.train_labels = tf.placeholder(tf.int32,shape=(batch_size, n_Transitions))
            self.test_inputs = tf.placeholder(tf.int32,shape=(n_Tokens))

            weights_input = tf.Variable(tf.random_normal([n_Tokens*embedding_size,hidden_size],stddev=0.1))
            embed = tf.reshape(tf.gather(self.embeddings, self.train_inputs),[batch_size,n_Tokens*embedding_size])
            biases_input = tf.Variable(tf.zeros(hidden_size))
            weights_output = tf.Variable(tf.random_normal([n_Transitions,hidden_size],stddev=0.1))

            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            condition = tf.equal(self.train_labels, -1)
            #case_true = tf.reshape(tf.zeros([Config.batch_size * parsing_system.numTransitions()], tf.float32),[Config.batch_size, parsing_system.numTransitions()]);

            case_true = tf.zeros([batch_size,n_Transitions],tf.int32)
            case_false = self.train_labels
            newLabels = tf.where(condition, case_true, case_false)

            l2 = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_output)+ tf.nn.l2_loss(biases_input)
            l2 = Config.lam / 2 * l2
            ce = tf.nn.softmax_cross_entropy_with_logits_v2(_sentinel=None,labels=newLabels,logits=self.prediction,dim=-1,name=None)

            self.loss = tf.reduce_mean(ce+l2)

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """
        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """
        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_inpu, weights_output):
        """
        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================
        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)
        =======================================================
        """

        hidden_layer = tf.pow(tf.add(tf.matmul(embed,weights_input),biases_inpu),3)
        output_layer = tf.matmul(weights_output,hidden_layer,transpose_a=False,transpose_b=True)

        return tf.transpose(output_layer)



def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):


    """
    =================================================================
    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)
    =================================================================
    """

    feature_list, merged_features = [], []

    s1,s2,s3 = c.getStack(0),c.getStack(1),c.getStack(2)
    b1,b2,b3 = c.getBuffer(0),c.getBuffer(1),c.getBuffer(2)

    lc1_s1, lc1_s2, rc1_s1, rc1_s2 = c.getLeftChild(s1,1), c.getLeftChild(s2,1), c.getRightChild(s1,1), c.getRightChild(s2,1)
    lc2_s1, lc2_s2, rc2_s1, rc2_s2 = c.getLeftChild(s1,2), c.getLeftChild(s2,2), c.getRightChild(s1,2), c.getRightChild(s2,2)

    lc1_lc1_s1, lc1_lc1_s2 = c.getLeftChild(lc1_s1,1), c.getLeftChild(lc1_s2,1)
    rc1_rc1_s1, rc1_rc1_s2 = c.getRightChild(rc1_s1,1), c.getRightChild(rc1_s2,1)

    feature_list.append(s1)
    feature_list.append(s2)
    feature_list.append(s3)
    feature_list.append(lc1_s1)
    feature_list.append(lc1_s2)
    feature_list.append(rc1_s1)
    feature_list.append(rc1_s2)
    feature_list.append(lc1_lc1_s1)
    feature_list.append(lc1_lc1_s2)
    feature_list.append(rc1_rc1_s1)
    feature_list.append(rc1_rc1_s2)

    count = 0

    for i in feature_list:

        count += 1
        merged_features.append(getWordID(c.getWord(i)))
        merged_features.append(getPosID(c.getPOS(i)))

        if count>6:
            merged_features.append(getLabelID(c.getLabel(i)))

    return merged_features



def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):

        if trees[i].isProjective():

            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)

    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)
