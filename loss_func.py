import tensorflow as tf
import numpy as np
import math as m

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    # Calculate the common term ({u_w}^T v_c) using matmul
    matmul_term = tf.matmul(true_w, inputs,  transpose_a=True)

    # Calculate the exponential term  exp({u_w}^T v_c)
    exp_term = tf.exp(matmul_term)

    A = tf.log(tf.add(1e-10, exp_term))  # based on FAQ suggestion added 1e-10 to avoid nans
    B = tf.log(tf.add(1e-10,tf.reduce_sum(exp_term, axis=1))) # based on FAQ suggestion added 1e-10 to avoid nans

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size]. shape=(128, 128), dtype=float32
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].  shape=(100000, 128) dtype=float32_ref
    biases: Biases for nce loss. Dimension is [Vocabulary, 1]. shape=(100000,) dtype=float32_ref
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].  shape=(128, 1), dtype=int32, 
    samples: Word_ids for negative samples. Dimension is [num_sampled]. len=64
    unigram_prob: Unigram probability. Dimesion is [Vocabulary]. len=100000

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================

    """
    print("---------------------------Shapes and Type of all input params --------------------")
    print("inputs >> ", inputs)
    print("weights >> ", weights)
    print("biases >> ", biases)
    print("labels >> ", labels)
    print("sample >> ", len(sample))
    print("unigram_prob >> ", len(unigram_prob))
    print("-----------------------------------------------------------------------------------")

    #equation term 1 for labels

    #solve for log(k.Pr(wo))
    # convert unigram prob input to tensor and get the probabilities corresponding to the labels and then flatten

    tf_unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    Prob_wo = tf.gather(tf_unigram_prob , labels) 
    Prob_wo = tf.reshape(Prob_wo, [-1]) # flatten
    print("Prob_wo >> ",Prob_wo)
    
    # multiply with 'k' and get log.
    k = float(len(sample))
    log_k_Prob_wo = tf.log(tf.add(1e-10 , tf.scalar_mul(k, Prob_wo))) # based on FAQ suggestion added 1e-10
    print("log_k_Prob_wo >> ", log_k_Prob_wo)
    

    # solve for sigma(s(wo,wc)) = sigma((uTc.uo) + bo)
    # solve for biases of outer words
    bo = tf.reshape(tf.nn.embedding_lookup(biases, labels), [-1]) # flatten after lookup
    print("bo >> ",bo)

    #get nce weights for labels
    wo = tf.reshape(tf.nn.embedding_lookup(weights, labels), [-1, labels.shape[0]]) # reshape after lookup for matmul
    print("weights_o 1st", wo)
    
    #get scores for outer words
    score_o = tf.nn.bias_add(tf.matmul(inputs, wo, transpose_b=True), bo) 
    print("score_o 1st", score_o)

    # final first term  - get the final first term translating the equation to corresponding tensorflow equivalent semantics
    term1 = tf.log(tf.add(1e-10,tf.sigmoid(tf.subtract(score_o, log_k_Prob_wo)))) # based on FAQ suggestion added 1e-10
    print("term1 >> ", term1)
    

    #equation term2 for negative samples

    #solve for log(k.Pr(wx))
    Prob_wx = tf.gather(tf_unigram_prob , sample) 
    print("Prob_wx >> ",Prob_wx)
    log_k_Prob_wx = tf.log(tf.add(1e-10,tf.scalar_mul(k, Prob_wx))) # based on FAQ suggestion added 1e-10
    print("log_k_Prob_wx >> ", log_k_Prob_wx)
     
    # solve for sigma(s(wx,wc)) = sigma((uTc.ux) + bo)
    #biases
    bx = tf.nn.embedding_lookup(biases, sample)
    bx = tf.reshape(bx, [-1])
    print("bx >> ",bx)

    #get nce weights for samples
    wx = tf.nn.embedding_lookup(weights, sample)
    print("weights_x 1st", wx)

    # get scores
    score_x = tf.nn.bias_add(tf.matmul(inputs, wx, transpose_b=True), bx) 
    print("score_x 1st", score_x)

    # get final term2 - get the final second term translating the equation to corresponding tensorflow equivalent semantics
    term2 = tf.reduce_sum(tf.log(tf.add(1e-10,tf.subtract(1.0 , tf.sigmoid(tf.subtract(score_x, log_k_Prob_wx))))), axis=1)  # based on FAQ suggestion added 1e-10
    print("term2 >> ", term2)

    # objective function
    finalCost = tf.scalar_mul(-1.0 , tf.add(term1, term2))
    print("finalCost >> ", finalCost)
    return finalCost
    