# Wordle Solver and Evaluator
Final project for 15418/618. Songyu Han and Joel Ye

[Project Page](https://joel99.github.io/parallel_final/)

**Summary**: We will develop a high throughput [Wordle](https://www.nytimes.com/games/wordle/index.html) solver based on [information theory heuristics](https://www.youtube.com/watch?v=v68zYyaEmEA), evaluating performances on GHC and PSC machines.

## Background
Wordle is a game with the goal of guessing a 5 letter English word in as few guesses as possible. After each guess, you are told which letters in your guess match letters in the solution, and if they are also in the right position in the word. Below, we show a three-round game.
<img width="656" alt="image" src="https://github.com/joel99/parallel_final/assets/14226466/a4f29eb2-c187-4e87-924f-cdbfdb474f9c">

The basic strategy we will implement is to minimize the average-case number of guesses for words drawn from a common word bank (1-10K words). This strategy is elaborated in this [3B1B video](https://www.youtube.com/watch?v=v68zYyaEmEA), which alternates between two steps for a given game:
2. For all guesses, compute the expected information gain \\( I(g) \\) for a guess, which is equal to the entropy of the feedback under the candidate answer distribution.
3. Identify the guess with the best expected information gain. Loosely, guessing the word where we are least certain of the feedback provides the most useful clue for future moves.
This 1-step heuristic is sufficient by itself (a V1 strategy) but also supports a more sophisticated policy (V2) that chooses words based on their probability of correctness at the current iteration and a value function which maps the empirical future guesses needed based on \\( f(I(g)) \\). There may be an opportunity to improve this value function with value iteration if we can support high throughput solver evaluation.

## The challenge
We focus largely on maximizing the speed of the V1 solver evaluation, where the parallelism challenges are greatest.
There are four nested layers of loops to run one evaluation of a strategy.
1. The evaluation loop (for answer in test bank, O(1K))
2. Turn loop (for turn in game, O(10))
3. Guess loop (for guess in legal guesses, O(10K))
4. Candidate answer loop (for answer in candidate answers, get feedback entropy, O(10K0)

In terms of a parallel programming framework, we have the following dependencies and parallelization opportunities:
1. Evaluation loop is data-parallel, (though there is repeated work for answers with shared letters).
2. The turn loop is iterative. Each iteration has successively smaller workloads (see below), which may motivate different parallelization strategies.
3. and 4. can be summarized as a [**scatter-reduce**](https://pytorch.org/docs/stable/generated/torch.scatter_reduce.html) operation from a prior distribution vector on answers to a probability vector of different feedback colorings. Let's break this step down:

Consider the following diagram that assumes a precomputed word compatibility matrix `color[guess][candidate]`. This word matrix is at a glance of ~size 10K x 10K, and contains color codes from `0-3^5`, i.e. 100Mb. The probability prior vector is a ~10Kb vector.

![image](https://github.com/joel99/parallel_final/assets/14226466/f286778c-10e7-4216-8fe1-fa3f67aa2dc5)


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
        probs.zero_()
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
- The game state over multiple rounds changes the number of candidate answers and thus requires **dynamic work redistribution** if we parallelized over candidates at all.
- Larger problems may require approximate solves to be tractable, which offers the opportunity to test the correctness degradation when we do approximate scatters with less synchronization.
- There are implementation uncertainties e.g. : Partitioning a large precomputed coloring matrix may not be beneficial vs recomputing string color matches on the fly.

The board's modest size makes this problem size a bit small for parallel computation on large machines, but this can be remedied by allowing larger vocabularies (up to 7 letter words, which allows about 60K possibilities and 3^7 bins, a 7Gb board) or more boards at once (as in [Qwordle](https://qwordle.bhat.ca/)) or [Quordle](https://www.merriam-webster.com/games/quordle/#/); we can evaluate scaling with problem size in either sense.

## Resources

We will use the accompanying [code for the 3B1B video](https://github.com/3b1b/videos/blob/master/_2022/wordle/simulations.py) as a reference for algorithmic correctness, and his video overall for guidance. However, his implementation is a naive for loop in Python. We will use OpenMP on GHC and move to PSC for late stage testing; the main implementation will likely depend on a shared memory abstraction. Time permitting, we may compare with a MPI based implementation where workers are dedicated to different subsets of candidates.

## Goals and Deliverables

**Plan to Achieve:**
Minimally, we expect to:
- Provide reference implementations: a serial CPU solver, and a vectorized python level implementation following above pseudocode that runs on either CPU or GPU.
- Implement and profile guess-loop, candidate-loop, and hybrid spatial parallelism of the scatter reduce operation. Parallelize with OpenMP and MPI on CPU, and CUDA on GPU.
- Analysis: Compare strategies for storing/partitioning precomputed coloring matrices, or computing colorings on the fly.
- Analysis: Characterize changing workloads in different guess iterations, and develop a work redistribution strategy to account for these.
- Benchmark performance on 6/7 letter wordle variants.

**Hope to Achieve:**
Our stretch goals are to consider extensions of the Wordle problem that add deeper parallelism.
- Extend and benchmark performance on multi-board variants (adds a board level parallelism loop)
- Implement the V2 solver that integrates an external value function (adds outer value function learning loop, where value function maps from information gain to expected score).
- Provide functionality to perform 2-step or deeper search instead of a 1-step heuristic, [described here](https://www.youtube.com/watch?v=fRed0Xmc2Wg).

**Deliverables at Poster Day:**
- We will provide visualizations of our different parallelism strategies and performance charts.

## Platform Choice

We will test both CPU and GPU parallelism, but will restrict to the GHC machines. The problem sizes we consider are not very large; if need be we will add experiments on the PSC machiens.

## Schedule
- 4/12:
  - Provide a serial CPU C++ and pytorch (python with C++ bindings) implementation of the V1 algorithm.
  - Add naive OpenMP parallelism on guess and candidate loops.
  - Provide test-bank evaluator and profile the two implementations on 5-letter wordle.
- 4/15 (4/16 milestone report)
  - Implement and profile guess level and candidate level parallelism while optimizing for memory locality and minimizing scatter reduce contention.
  - Evaluate correctness degradation with reduced synchronization on reduction.
  - Experiment with coloring matrix partitioning or on the fly coloring computation (all in OpenMP).
- 4/22
  - Profile and optimize workload balancing across turns.
  - Scale problem size in number of letters (up to 7)
  - Implement CUDA approach.
- 4/29
  - Implement message-passing solver with MPI
  - Maybe: extend problem size in number of boards to solve.
- 5/5  
  - Writing up report and preparing poster.

