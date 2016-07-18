#!/usr/bin/env python2.7

import numpy as np

# Implementation of Linear UCB with a stepwise reset of the model (on a per article basis). If an article
# is available for more than 24 hours its model is reset. This leads currently to the best score on the evaluation system.
class LinUCB:
    all_articles = []
    all_M = {}
    all_M_inv = {}
    all_b = {}
    all_w = {}
    alpha = 0.2
    current_article = None  # current recommendation
    current_user = None  # user for which the article was recommended

    def set_articles(self, articles):
        self.all_articles = articles
        self.counter = 0
        # initialize M and b for each article:
        for article in self.all_articles:
            self.all_M[article] = np.identity(6)
            self.all_b[article] = np.zeros((6, 1))
            self.all_M_inv[article] = np.identity(6)
            self.all_w[article] = np.zeros((6, 1))

    def resetArticle(self, article):
        self.all_M[article] = np.identity(6)
        self.all_b[article] = np.zeros((6, 1))
        self.all_M_inv[article] = np.identity(6)
        self.all_w[article] = np.zeros((6, 1))

    def ucb(self, article, user, timestamp):
        M_inv = self.all_M_inv[article]
        w = self.all_w[article]
        ucb = np.dot(w.T, user) + self.alpha * np.sqrt(np.dot(user.T, np.dot(M_inv, user)))
        return ucb


    def recommend(self, timestamp, user_features, articles):
        user_features = np.reshape(user_features, (6, 1))
        best_ucb = -np.inf
        for article in articles:
            if article not in self.all_M: continue
            current_ucb = self.ucb(article, user_features, timestamp)
            if current_ucb > best_ucb:
                best_ucb = current_ucb
                self.current_article = article
        self.current_user = user_features
        return self.current_article


    def update(self, reward):
        if reward != -1:
            self.counter += 1
            article = self.current_article
            user = self.current_user
            self.all_M[article] += np.dot(user, user.T)
            self.all_b[article] += reward * user
            # precompute M^-1 and w for UCB
            self.all_M_inv[article] = np.linalg.inv(self.all_M[article])
            self.all_w[article] = np.dot(self.all_M_inv[article], self.all_b[article])



linucb = LinUCB()

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    linucb.set_articles(art)


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    linucb.update(reward)


# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    return linucb.recommend(timestamp, user_features, articles)
