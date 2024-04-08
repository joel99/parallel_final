# Wordle Solver and Evaluator
Final project for 15418/618. Songyu Han and Joel Ye

[Project Page](https://joel99.github.io/parallel_final/)

**Summary**: We will develop a high throughput [Wordle](https://www.nytimes.com/games/wordle/index.html) solver based on [information theory heuristics](https://www.youtube.com/watch?v=v68zYyaEmEA), evaluating performances on GHC and PSC machines.

## Background
Wordle is a game with the goal of guessing a 5 letter English word in as few guesses as possible. After each guess, you are told which letters in your guess match letters in the solution, and if they are also in the right position in the word. Below, we show a three-round game.
<img width="656" alt="image" src="https://github.com/joel99/parallel_final/assets/14226466/a4f29eb2-c187-4e87-924f-cdbfdb474f9c">

A basic strategy is to minimize the average-case number of guesses for words drawn from a common word bank (1-10K words). This strategy is elaborated in this [3B1B video](https://www.youtube.com/watch?v=v68zYyaEmEA), and has three components:
1. Compute the expected feedback entropy with respect to possible candidate answer words (probabilities drawn from a lookup table), given state of board.
2. Compute the expected information gain \\( I(g) \\) for a guess, which is equal to the entropy of the feedback given under the answer distribution.
3. Identify the guess with the best expected information gain. Loosely, guessing the word where we are least certain of the feedback provides the most useful clue for future moves.
This 1-step heuristic is sufficient by itself (a V0 strategy) but also supports a more sophisticated policy (V1) that chooses words based on their probability of correctness at the current iteration and a value function which maps the empirical future guesses needed based on \\( f(I(g)) \\).

Proposed strategies like this one must be evaluated on a test set of answers. This evaluation is typically treated as data-parallel across answers. There may be a chance to improve the heuristic policy with value iteration if the evaluator is fast enough.

## The challenge

There are four nested layers of loops to run one evaluation of a strategy.
1. The evaluation loop (for answer in test bank, O(1K))
2. Turn loop (for turn in game (O(10)))
3. Guess loop (for answer in legal guesses, which is best next move?)
4. Candidate answer loop (get feedback entropy by considering probabilities of the unknown, different answers)

Re-evaluate in a parallel programming framework, we have the following dependencies:
1. Evaluation loop is data-parallel, though there is repeated work for answers with shared letters
2. The turn loop is iterative; optimistic computation of future loops unlikely to be helpful. However, each iteration changes the workload by changing the probability 
3. and 4. can be summarized as a [**scatter-reduce**](https://pytorch.org/docs/stable/generated/torch.scatter_reduce.html) operation from our prior distribution vector on answers to a probability vector of different feedback colorings. Let's break this step down:

Consider the following diagram that assumes a precomputed word compatibility matrix `color[guess][candidate]`. This word matrix is at a glance of ~size 10K x 10K, and contains color codes from `0-3^5`, i.e. 100Mb. The probability prior vector is a ~10Kb vector.


![image](https://github.com/joel99/parallel_final/assets/14226466/61946a9c-7b13-4c3c-88a8-56884cfea590)


---

The pytorch pseudocode might be:
```
def solve(coloring: torch.Tensor, prior: torch.Tensor, answer: str | int, rounds=6):
    r"""
        coloring: tensor of size ([Word, Word])
        prior: tensor of size ([Word])
    """
    probs = torch.zeros(len(coloring.unique()))
    for round in range(rounds):
        guess_feedback_entropy = torch.zeros(Word)
        for guess in range(coloring.size(0)):
            guess_labels = coloring[guess]
            probs.scatter_reduce(0, index=guess, src=prior, reduce='sum')
            entropy = probs.to_entropy()
            guess_feedback_entropy[guess] = entropy
        feedback = compute_feedback(guess_feedback_entropy.argmax(), answer)
        update_coloring_matrix(guess, feedback) # update board state, either subset or zero out
```
.

The parallelism challenge is an efficient implementation of iterated scatter reduce on potentially large matrices (O(0.1-5)Gb).
- A single round of scatter reduce can be implemented by _parallelizing along the guess and/or candidate dimensions_, with uncertain potential for optimizing color matrix and probability vector access patterns on cache line pulls. The scatter operation may be a bottleneck as it involves many workers writing to a single probability vector. This requires understanding of **memory locality** and **write contention**.
- The maintained state over multiple rounds changes the number of viable candidates and thus could benefit from **work redistribution** if we parallelized over candidates at all. (May depend on how we update matrix)
- Larger problems may require approximate solves to be tractable, which offers the opportunity to test the correctness degradation when we do approximate scatters with less synchronization.
- There are implementation uncertainties e.g. : Partitioning a full large matrix may not be beneficial vs recomputing string color matches on the fly.

(This hits all the criteria for being a superset problem compared to the wire-routing homework assignments.)

The board's modest size makes this problem size a bit small for parallel computation on large machines, but this can be remedied by allowing larger vocabularies (up to 7 letter words, which allows about 60K possibilities and 3^7 bins, a 7Gb board) or more boards at once (as in [Qwordle](https://qwordle.bhat.ca/)) or [Quordle](https://www.merriam-webster.com/games/quordle/#/); we can evaluate scaling with problem size in this sense.


## Resources

We will use the accompanying [code for the 3B1B video](https://github.com/3b1b/videos/blob/master/_2022/wordle/simulations.py) as a reference for algorithmic correctness, and his video overall for guidance. However, his implementation is a naive for loop in Python. We will use OpenMP on GHC and move to PSC for late stage testing; the main implementation will likely depend on a shared memory abstraction. Time permitting, we may compare with a MPI based implementation where workers are dedicated to different subsets of candidates.

## Goals and Deliverables

**Plan to Achieve:**
Minimally, we expect to:
- Provide an serial implementation that reproduces the average score of 3B1B's wordle solver, and a vectorized python level implementation following above pseudocode. These should provide performance references on CPU/GPU.
- Explore guess-loop, candidate-loop, and hybrid spatial parallelism on the scatter reduce operation with CPU thread level parallelism on base wordle game.
- Characterize changing workload requirements in different guess iterations and develop a work redistribuiton strategy based on these.
- Provide a handwritten CUDA implementation of the solver (GHC?).
- Benchmark performance on 6/7 letter wordle variants.
- Test strategy for preserving probability computation across rounds (tentative?)

**Hope to Achieve:**
- Extend and benchmark performance on multi-board variants.
- Demonstrate that accelerated solving enables an outer loop optimization of the empirical value function by evaluating against the test set.
- Provide functionality to perform 2-step or deeper search instead of a 1-step heuristic, [described here](https://www.youtube.com/watch?v=fRed0Xmc2Wg).

**Deliverables at Poster Day:**
- Most likely results will simply be performance charts. We may build a visualizer showing our solver's progress, but feel that'd be an orthogonal goal to the main study (there's already a video of the algorithm).

## Platform Choice

We will begin with a low-worker count parallelism to get a sense of the problem scale. It may not be sufficiently large to justify many more cores or a GPU implementation (since the naive approach was suitable for the resource video.) If we do not get a satisfactory speedup, a GPU implementation would be interesting as the innnermost string comparison is a heterogeneous operation, and would perhaps motivate using a big lookup table instead of computing it on the fly.

## Schedule
- 4/12:
  - provide a serial and pytorch implementation of the V0 algorithm with word frequencies integrated.
  - provide test-bank evaluator and profile the two implementations.
- 4/15 (4/16 milestone report)
  - implement and profile base passes on guess level and candidate level parallelism, using shared memory and OpenMP
  - experiment with different memory partitioning strategies
- 4/22
  - add workload rebalancing across turns
  - scale problem size in number of letters in words 
  - implement CUDA solution, targeting candidate parallelism or both (since many threads to work with now)
- 4/29
  - implement message-passing solver with MPI
  - Maybe: extend problem size in number of boards to solve.
- 5/5  
  - Writing up report and preparing poster.

