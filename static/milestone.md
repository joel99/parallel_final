# Project Checkpoint Report
[Project Page](https://joel99.github.io/parallel_final/)

In this checkpoint report, we will present our current progress on the term project as well as some new discoveries we have realized as we were working on our current implementations. We will also discuss potential updates in our project schedule in this report.

## Current Progress

We began working on this project after our revised proposals are completed, and we have successfully implemented a sequential solver that replicates the majority of the functionalities outlined in [_3Blue1Brown_'s video](https://www.youtube.com/watch?v=v68zYyaEmEA). In addition to providing a fully functional sequential implementation, we have embedded automated testing and timing code into our program to facilitate performance evaluation in parallel implementations.

Our sequential algorithm roughly follows the routines underlined below: (Some edge cases and implementational details omitted) The game loop may be repeated multiple times to evaluate the performance of wordle solver on different words.
```
<Initialization Sequence: Set up word list and prior weights>
    Precompte the "pattern matrix" containing feedbacks for all guess-answer pairs
    # Game Loop Begins
    while <game not complete>:
        # Computation Phase
        for guess_word in <word list>:
            Pool the total weights from word priors for each coloring pattern via **scatter reduce**
            Compute guess_word's expected entropy I(g) by normalizing the pooled weights into a probability distribution.
            record guess_word's score as a function of its expected entropy.
        # Candidate Selection Phase
        candidate_word = argmax(scores)
        Obtain feedback by submitting candidate_word as guess
        # Solver Update Phase
        for word in <word list>:
            set word's prior weight to 0 if feedback(candidate_word, word) does not match.
        compute the sum of all prior weights to be used in the normalization step
    # Game Loop Ends
```

We have also carefully analysed our wordle solver algorithm, and two of the most important operations to be parallelized in our algorithm are **Map Reduce** and **Scatter Reduce**. The reduction algorithm is relatively straight forward to parallelise, and many parallel computing frameworks, such as OpenMP and MPI, provides primitives to efficiently perform reductions. However, the Scatter Reduce method is characterized by write contention and memory access indirections, which makes it very tricky to parallelize. We have proposed 3 potential approaches to parallel scatter reduce to be experimented with: a lock based approach, a reduction based approach and a reformulation of the scatter reduce problem as a sparse matrix vector multiplication problem. We shall further evaluate the effectiveness of each approach by running each scatter reduce implementation on very large matrices and vectors.

As mentioned in the proposal document, during the computational phase, we could either parallelize over the "guess_word" dimension ("across-word parallelism"), which we believe is the natural approach to parallelize our solver, or experiment with "within-word parallelism" by incorporating our parallel scatter reduce. We are also interested in incorporating SIMD to further enhance the performance of our CMU solver. In the following weeks, we are also interested in completing a CUDA implementation of our wordle solver. Although the pattern matrix pre-computation phase is massively data parallel, there will be challenges in our actual solver loop routine, as CUDA does not have natural reduction primitives and has very limited synchronization constructs. This forces us to reconsider our current scatter reduce and map reduce implementations.

We have also decided modify the overall directive of this project. As mentioned in [_3Blue1Brown_'s addendum to their wordle video](https://www.youtube.com/watch?v=fRed0Xmc2Wg), most of the algorithmic refinemenets included in his version 2 solver are aimed at reducing the average rounds required for a correct guess. However, since we are mostly interested in improving and measuring the parallel performance of the wordle solver on different machine architectures, it is likely that we will sideline the inclusion of some features of _3Blue1Brown_'s algorithm in our parallel implementations. In the "Updated Project Goals and Schedules" section, we have modified our objectives to reflect this change of project directive.

## Deliverable at Poster Session

## Preliminary Testing Results
TODO

## Concerns and Unknowns

Our current progress mostly aligns with our initial expectations and it is likely that we will be able to complete all the "plan to achieve" deliverable items. However, due to the delays in the project proposal phase, it is expected for us to perform most of the parallel developmental work in the next two weeks and spend the last week conducting parallel program experimentations.

One preliminary result item worth mentioning is that we are somewhat underwhelmed by the performance of parallel scatter reduce. On one of the team member's computer (Macbook Pro 16 inch, 8 Core i9-9880H CPU), all of our current parallel scatter reduce implementations perform significantly wrose than our sequential baseline on the typical input sizes used in the wordle solver (10 to 20 thousand input items, and an output dimension of 243), and the performance gain due to parallelism on very large data sets is also very limited. We propose improving the locality of memory access by sorting the input arrays, but we are especially concerned as sorting is a task that requires more work and memory writies than the scatter reduce operation itself.




## Updated Project Goals and Schedules
- Week 4/12:
  - 🔴 Provide a serial CPU C++ and pytorch (python with C++ bindings) implementation of the V1 algorithm. (Completed, SH & JY)
  - 🔴 Analyze sequential algorithm and determine multiple parallel appraoches to the Wordle solver. (Completed, SH & JY)
  - 🔴 Add naive OpenMP parallelism on guess and candidate loops. (Completed, SH)
  - 🔴 Provide test-bank evaluator and profile the two implementations on 5-letter wordle. (Completed, SH & JY)
- Week 4/15 (First Half):
  - 🔵 Implement and profile guess level and candidate level parallelism while optimizing for memory locality and minimizing scatter reduce contention. (In Progress, SH & JY)
  - 🟢 Evaluate performance characteristics of various implementations of scatter reduce and map reduce. (New Task, In Progress, SH)
- Week 4/15 (Second Half):
  - ⚫ Evaluate correctness degradation with reduced synchronization on reduction. (Pending, JY)
  - ⚫ Implement message-passing solver with MPI (Pending, SH & JY)
  - ⚫ Start to implement GPU solver in CUDA (Pending, SH & JY)
- Week 4/22 (First Half):
  - 🔵 Experiment with coloring matrix partitioning or on the fly coloring computation (all in OpenMP). (In Progress, JY)
  - ⚫ Profile and optimize workload balancing across turns. (Pending, SH & JY)
- Week 4/22 (Second Half):
  - ⚫ Continue Optimizing the CUDA Solver (Pending, SH)
  - 🔵 Scale problem size in number of letters (up to 7) (In Progress, JY)
- Week 4/29 (First Half):
  - ⚫ Profile and analyze performance characteristics of GPU solver (Pending, SH & JY)
  - 🟢 Perform problem size sensitivity analysis (New Task, Pending, SH)
  - ⚫ Hope to Achieve: extend problem size in number of boards to solve. (Pending, SH & JY)
- Week 4/29 (Second Half): 
  ⚫ Writing up report and preparing poster. (Pending, SH & JY)
