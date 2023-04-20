The training data file has the same format as the trial data file:
<ID><tab><Target><tab><Tweet><tab><Stance>

where
<ID> is an internal identification number;
<Target> is the entity of interest (e.g., "Hillary Clinton"; there are 5 different targets in the training data);
<Tweet> is the text of a tweet;
<Stance> is the stance label.

The possible stance labels are:
1. FAVOR: We can infer from the tweet that the tweeter supports the target (e.g., directly or indirectly by supporting someone/something, by opposing or criticizing someone/something opposed to the target, or by echoing the stance of somebody else).
2. AGAINST: We can infer from the tweet that the tweeter is against the target (e.g., directly or indirectly by opposing or criticizing someone/something, by supporting someone/something opposed to the target, or by echoing the stance of somebody else).
3. NONE: none of the above.

The possible targets are:
1. Atheism
2. Climate Change is a Real Concern
3. Feminist Movement
4. Hillary Clinton
5. Legalization of Abortion

Note: Each of the instances in the training data has an additional hashtag (#SemST) that just marks that the tweet is part of the SemEval-2016 Stance in Tweets shared task. Your systems are free to delete this hashtag during pre-processing or simply ignore it. Human annotators of stance did not see this hashtag in the tweet when judging stance.


