# Wordle Solver and Evaluator
Final project for 15418/618. Songyu Han and Joel Ye

[Project Page](https://joel99.github.io/parallel_final/)

**Summary**: We will implement a [Wordle](https://www.nytimes.com/games/wordle/index.html) solver based on [information theory heuristics](https://www.youtube.com/watch?v=v68zYyaEmEA), evaluating performances on GHC and PSC machines.

## Background
Wordle is a game with the goal of guessing a 5 letter English word in as few guesses as possible. After each guess, you are told which letters in your guess match letters in the solution, and if they are also in the right position in the word. Below, we show a three-round game.
<img width="656" alt="image" src="https://github.com/joel99/parallel_final/assets/14226466/a4f29eb2-c187-4e87-924f-cdbfdb474f9c">

A basic strategy is to minimize the average-case number of guesses for words drawn from a common word bank (1-10K words). This strategy is elaborated in this [3B1B video](https://www.youtube.com/watch?v=v68zYyaEmEA), and has three components:
1. Compute the entropy of candidate answer words (probabilities drawn from a lookup table), given state of board.
2. Compute the expected information gain \\( I(g) \\) for a guess, which is equal to the entropy of the feedback given under the answer distribution.
3. Identify the guess with the best expected information gain. Loosely, guessing what we are least sure the feedback will be provides the most useful clue for future moves.
This 1-step heuristic supports an approximate policy that chooses words based on their probability of correctness at the current iteration and a value function which maps the empirical future guesses needed based on \\( f(I(g)) \\).

Further, proposed strategies like this one must be evaluated on a test set of answers. This evaluation is typically treated as data-parallel across answers. There may be a chance to improve the heuristic policy with value iteration if the evaluator is fast enough.

## The challenge

There are three nested layers of loops, each which work with about 1-10K words.
1. The evaluation loop (for answer in test bank)
2. Guess loop (for answer in legal guesses, which is best next move?)
  - We can experiment with the value of per-worker minibatch aggregation before making a comparison across workers.
3. Candidate answer loop (get probability of candidate by weighing lookup probability with compatibility with guess)
  - The working set of candidates is dynamically sized based on the state of the board, hence there's a workload balancing challenge at the innermost loop.
  - Computing  \\( p(x) \\) for different candidates have no interdependency, but probabilities must be synchronized and gathered to compute the result.

This work is characterized by a high communication to computation ratio; the computation is effectively a few multiplications and max operations.

At minimum, we expect to try parallelization on the inner two loops over different guesses and different candidates. There are a number of implementation details that are interesting for a parallel programming class. For example, parallelizing the guess loop (2) may have memory locality benefits when a single worker compares against multiple candidates, but the same can be said for parallelizing over candidates for all possible guesses. Further, the workload profile shifts over different rounds of guessing, because the candidate pool should exponentially decrease. This may motivate different distribution of work in the inner loop, or staggered work in the outer loop.


## Resources

We will use the accompanying [code for the 3B1B video](https://github.com/3b1b/videos/blob/master/_2022/wordle/simulations.py) as a reference for algorithmic correctness, and his video overall for guidance. However, his implementation is a naive for loop in Python. We will use OpenMP on GHC and move to PSC for late stage testing; the main implementation will likely depend on a shared memory abstraction. Time permitting, we may compare with a MPI based implementation where workers are dedicated to different subsets of candidates.

## Goals and Deliverables

**Plan to Achieve:**
Minimally, we expect to:
- Provide an serial implementation that reproduces the average score of 3B1B's wordle solver.
- Compare the performance of a serial solvers with CPU-based thread parallelism. Explore guess-loop vs candidate-loop parallelism.
- Characterize changing workload requirements in different guess iterations and propose a hybrid guess-loop / candidate-loop strategy.

We also feel it's likely we will be able to provide a CUDA implementation of this solver, but it's unclear whether this problem would benefit from high-thread count parallelism.

**Hope to Achieve:**
- Demonstrate that accelerated solving enables an outer loop optimization of the empirical value function by evaluating against the test set.
- Provide functionality to perform 2-step or deeper search instead of a 1-step heuristic, [described here](https://www.youtube.com/watch?v=fRed0Xmc2Wg).
- If we implement a CUDA functionality, it'd be interesting to compare our low level algorithm against the high level language that e.g. Pytorch provides.

**Deliverables at Poster Day:**
- Most likely results will simply be performance charts. We may build a visualizer showing our solver's progress, but feel that'd be an orthogonal goal to the main study (there's already a video of the algorithm).

## Platform Choice

We will begin with a low-worker count parallelism to get a sense of the problem scale. It may not be sufficiently large to justify many more cores or a GPU implementation (since the naive approach was suitable for the resource video.) If we do not get a satisfactory speedup, a GPU implementation would be interesting as the innnermost string comparison is a heterogeneous operation, and would perhaps motivate using a big lookup table instead of computing it on the fly.

## Schedule
- 4/2:
  - provide a serial implementation with uniform prior over words. (Stretch: Integrate external word frequency prior)
- 4/8:
  - provide test-bank evaluator and gather statistics about serial solve times.
  - implement base passes on guess level and/or candidate level parallelism.
- 4/15 (4/16 milestone report)
  - optimize (e.g. load balance, minimize communication, local aggregation) and profile the two types of parallelism; compare against serial in report.
- 4/22
  - profile workload variability across different iterations of a solve and propose an adaptive parallelism strategy OR
  - implement CUDA solution, targeting candidate parallelism or both (since many threads to work with now)
- 4/29
  - identify limiting factors in current solutions  
  - experiment with algorithmic improvements: outer-loop optimization of value function, 2+ depth search.
  - experiment with parallelization improvements: attempts to minimize redundant calculation
- 5/5  
  - Writing up report and preparing poster.

