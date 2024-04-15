# Project Checkpoint Report
[Project Page](https://joel99.github.io/parallel_final/)

In this checkpoint report, we will present our current progress on the term project as well as some new discoveries we have realized as we were working on our current implementations. We will also discuss potential updates in our project schedule in this report.

## Current Progress

We began working on this project after our revised proposals are completed, and we have successfully implemented a sequential solver that replicates the majority of the functionalities outlined in [_3Blue1Brows_'s video](https://www.youtube.com/watch?v=v68zYyaEmEA). In addition to providing a fully functional sequential implementation, we have embedded automated testing and timing code into our program to facilitate performance evaluation in parallel implementations.

Our sequential algorithm roughly follows the routines underlined below: (Some edge cases and implementational details omitted) The game loop may be repeated multiple times to evaluate the performance of wordle solver on different words.
```
<Initialization Sequence: Set up word list and prior weights>
  Precompte the "pattern matrix" containing feedbacks for all guess-answer pairs
  ** Game Loop Begins **
  while <game not complete>:
    **Computation Phase**
    for guess_word in <word list>:
        Pool the total weights from word priors for each coloring pattern via **scatter reduce**
        Compute guess_word's expected entropy I(g) by normalizing the pooled weights into a probability distribution.
        record guess_word's score as a function of its expected entropy.
    **Word Selection Phase**
    candidate_word = argmax(scores)
    Obtain feedback by submitting candidate_word as guess
    **Update Phase**
    for word in <word list>:
        set word's prior weight to 0 if feedback(candidate_word, word) does not match.
    compute the sum of all prior weights to be used in the normalization step
  ** Game Loop Ends **
```




- What we have learned
- What we have implemented
- Different implementational decisions
- Different levels of parallelism
- Discuss

In addition to providing a correct sequential implementation that correctly replicates most of the behaviors of the 

However, we also decided modify the overall directive of this project.

## Preliminary Testing Results

## Concerns and Unknowns
Our current progress mostly aligns with our initial expectations and it is likely that we will be able to complete all the "plan to achieve" deliverable items. However, due to the delays experienced in the project proposal phase, it is expected for us to perform most of the parallel program developmental work in the following two weeks.




## Updated Project Goals and Schedules
- 4/12:
  - 🔴 Provide a serial CPU C++ and pytorch (python with C++ bindings)     implementation of the V1 algorithm. (Completed)
  - 🔴 Analyze sequential algorithm and determine multiple parallel appraoches to the Wordle solver. (Completed)
  - 🔵 Add naive OpenMP parallelism on guess and candidate loops. (In Progress)
  - 🔴 Provide test-bank evaluator and profile the two implementations on 5-letter wordle. (Completed)
- 4/15 (4/16 milestone report)
  - 🔵 Implement and profile guess level and candidate level parallelism while optimizing for memory locality and minimizing scatter reduce contention. (In Progress)
  - ⚫ Evaluate correctness degradation with reduced synchronization on reduction. (Pending)
  - 🟢 Evaluate performance characteristics with multiple implementations of scatter reduce and map reduce (New Task, In Progress)
  - ⚫ Implement message-passing solver with MPI (Pending)  
- 4/22
  - ⚫ Experiment with coloring matrix partitioning or on the fly coloring computation (all in OpenMP). (Pending)
  - ⚫ Profile and optimize workload balancing across turns. (Pending)
  - 🔵 Scale problem size in number of letters (up to 7) (In Progress)
  - ⚫ Implement and optimize GPU implementation of the wordle solver in CUDA (Pending)
- 4/29
  - ⚫ Hope to Achieve: extend problem size in number of boards to solve. (Pending)
  - 🟢 Perform problem size sensitivity analysis (New Task, Pending)  
- 5/5  
  ⚫ Writing up report and preparing poster. (Pending)
