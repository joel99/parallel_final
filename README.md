# Wordle Solver and Evaluator
Final project for 15418/618. Songyu Han and Joel Ye

URL: https://github.com/joel99/parallel_final

**Summary**: We will implement a [Wordle](https://www.nytimes.com/games/wordle/index.html) solver based on [information theory heuristics](https://www.youtube.com/watch?v=v68zYyaEmEA), evaluating performances on GHC and PSC machines.

## Background
Wordle is a game with the goal of guessing a 5 letter English word in as few guesses as possible. After each guess, you are told which letters in your guess match letters in the solution, and if they are also in the right position in the word. Below, we show a three-round game.
<img width="656" alt="image" src="https://github.com/joel99/parallel_final/assets/14226466/a4f29eb2-c187-4e87-924f-cdbfdb474f9c">

A basic strategy is to minimize the average-case number of guesses for words drawn from a common word bank (1-10K words). This strategy is elaborated in this [3B1B video](https://www.youtube.com/watch?v=v68zYyaEmEA), and has three components:
1. Compute the entropy of a distribution of words (probabilities drawn from a lookup table).
2. Compute the expected information gain for a candidate guess, which is equal to the entropy of the feedback given under the answer distribution.
3. Identify the guess with the best expected information gain. Loosely, guessing what we are least sure the feedback will be provides the most useful clue for future moves.
This 1-step heuristic supports an approximate policy that chooses words based on their probability of correctness at the current iteration and expected information gain if incorrect.

Further, proposed strategies like this one must be evaluated on a test set of answers. This evaluation is typically treated as data-parallel across answers. Technically, there may be a chance to improve the approximate policy with value iteration if the evaluator is fast enough.

## The challenge

The core parallelization question is building a search tree that can be parallelized across the evaluation of different guesses, and within the evaluation of an individual guesses.

There is further a serial multi-round decision-making structure.

## The challenge


## Resources


## Goals and Deliverables


## Platform Choice


## Schedule
