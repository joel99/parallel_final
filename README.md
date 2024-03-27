# Wordle Solver and Evaluator
Final project for 15418/618. Songyu Han and Joel Ye

URL: https://github.com/joel99/parallel_final

**Summary**: We will implement a [Wordle](https://www.nytimes.com/games/wordle/index.html) solver based on [information theory heuristics](https://www.youtube.com/watch?v=v68zYyaEmEA), evaluating performances on GHC and PSC machines.

## Background
Wordle is a game with the goal of guessing a 5 letter English word in as few guesses as possible. After each guess, you are told which letters in your guess match letters in the solution, and if they are also in the right position in the word. Below, we show a three-round game.
<img width="656" alt="image" src="https://github.com/joel99/parallel_final/assets/14226466/a4f29eb2-c187-4e87-924f-cdbfdb474f9c">

A basic strategy is to minimize the average-case number of guesses for words drawn from a common word bank (1-10K words). This strategy is elaborated in this [3B1B video](https://www.youtube.com/watch?v=v68zYyaEmEA), and has three components:
1. Compute the entropy of candidate answer words (probabilities drawn from a lookup table), given state of board.
2. Compute the expected information gain for a guess, which is equal to the entropy of the feedback given under the answer distribution.
3. Identify the guess with the best expected information gain. Loosely, guessing what we are least sure the feedback will be provides the most useful clue for future moves.
This 1-step heuristic supports an approximate policy that chooses words based on their probability of correctness at the current iteration and expected information gain if incorrect.

Further, proposed strategies like this one must be evaluated on a test set of answers. This evaluation is typically treated as data-parallel across answers. There may be a chance to improve the heuristic policy with value iteration if the evaluator is fast enough.

## The challenge

There are three nested layers of loops, each which work with about 1-10K words.
1. The evaluation loop (for answer in test bank)
2. Guess loop (for answer in legal guesses, which is best next move?)
  - We can experiment with the value of per-worker minibatch aggregation before making a comparison across workers.
3. Candidate answer loop (get probability of candidate by weighing lookup probability with compatibility with guess)
  - The working set of candidates is dynamically sized based on the state of the board.
  - Computing $p(x)$ for different candidates have no interdependency, but probabilities must be gathered to compute the result.

At minimum, we expect to try parallelization on the inner two loops over different guesses and different candidates. There are a number of implementation details that are interesting for a parallel programming class. For example, parallelizing the guess loop (2) may have memory locality benefits when a single worker compares against multiple candidates, but the same can be said for parallelizing over candidates for all possible guesses. Further, the workload profile shifts over different rounds of guessing, because the candidate pool should exponentially decrease. This may motivate different distribution of work in the inner loop, or staggered work in the outer loop.


## Resources


## Goals and Deliverables


## Platform Choice


## Schedule
