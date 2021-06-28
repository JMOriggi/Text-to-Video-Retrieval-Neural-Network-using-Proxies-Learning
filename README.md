# Text-to-Video-Retrieval-Neural-Networkvia-using-Proxies-Learning

## Overview
We address the problem of retrieving a specific moment
from a video by a natural language query. This is a challenging
problem because a target moment may have semantic
relations to other temporal moments in the untrimmed
video. Existing methods have some approaches to tackle
this challenge by modeling the temporal relations between
video moments. In this paper, we propose a novel model
where we make use of proxies that are learnable parameters
represents some high level characteristics. Our proposed
approach is capable of learning faster with a better accuracy.
We evaluate the proposed model on two challenging
benchmarks, i.e., Charades-STA and ActivityNet Captions
where our model has promising results.

## Moment localization in video using textual queries
Localizing video moments that
match to a query sentence is a challenging task. It requires
both understanding of language query and video. In addition
to that it requires comprehending the relationship between
vision and language. Recently, significant progress
has been achieved at video grounding task.
Current methods can be grouped into two categories as
the ones use a two-stage pipeline and the ones use a onestage
pipeline. Most of the current language-queried video
grounding methods use the two-stage pipeline
where they first generate moment candidates and calculate
similarity scores between these candidates and query sentence.

![Alt text](/git-docs/2D_TAN.JPG) 

Reference paper [2D-TAN_paper_link](https://arxiv.org/pdf/1912.03590.pdf)

## Proxies NCA ++
The is a problem of distance metric
learning (DML), where the task is to learn an effective
similarity measure between images. There is a strategy that
tackles this problem via class distribution estimation. The
main motivation of this strategy is reducing computation by
comparing samples to proxies. Methods as an example of
this line of work is the Magnet Loss, ProxyNCA
and more lately ProxyNCA++ in which each sample is
assigned to a cluster centroid, and at each training batch,
samples are attracted to cluster centroids of similar classes
and pushed away by cluster centroids of different classes.

![Alt text](/git-docs/proxies_nca++.JPG) 

Reference paper [Proxies_paper_link](https://arxiv.org/pdf/1912.03590.pdf)

## Proxies integration with 2D-TAN model
We used the extracted mid-level features from the model
to compute the proxy-loss. Since the original proxy method
only computes the loss between one feature vector and all
the proxies, we modified the loss function in order to handle
the 144 clips representations. We aggregate the scores
of the distances between the proxies and the clip features,
compute the average, and re-scale the result by a coefficient
of 1/100. This last step, of re-scaling the proxy loss value
was necessary to compute the aggregated loss together with
the BCE loss used by the original 2D-TAN model (Figure
2). The loss value obtained by summing the 2 losses is than
used to back-propagate the error and update both the proxies
and 2D-TAN model weights.

![Alt text](/git-docs/proxies_loss.JPG) 

## Results
We have experimented different training configurations
in order to better understand the effect of the proxies on the
training speed and accuracy. As the Figure 3 shows we obtain
different behaviour depending on the batch size considered,
and depending on the integration of the proxies during
training. We used the mIoU as a metric to compare the results.
From these experiments, we deduced that the proxies
play a role as regularizer and catalyzer during the training
phase of the model. In fact when we consider the proxies
we obtained a maximum value in the 35.7-36.2 range mIoU
for the test set in the early stage of the training phase, while
by not including the proxies, we obtained a value in the 35-
35.6 range. From the results we
can also notice that the proxy model improves performances
also when compared to a training without proxies but with
higher batch size. This is particularly interesting
since it proves that the proxies are here not only to increase
the batch size but also to provides an high level video representation
that help the model learning process.

![Alt text](/git-docs/results.JPG) 

