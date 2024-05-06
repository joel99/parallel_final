# Project Checkpoint Report
[Project Page](https://joel99.github.io/parallel_final/)

In this checkpoint report, we will present our current progress on the term project as well as some new discoveries since our project proposals.

## Current Progress

We began working on this project after our revised proposals are approved, and we have successfully implemented a sequential solver that replicates the majority of the functionalities outlined in [_3Blue1Brown_'s video](https://www.youtube.com/watch?v=v68zYyaEmEA). In addition to providing a fully functional sequential implementation, we have embedded automated testing and profiling into our program to facilitate performance evaluation.

Our sequential algorithm roughly follows the routines underlined below: (Some edge cases and implementational details omitted) The game loop is repeated for each word in the test set, if provided.
```
<Initialization Sequence: Set up word list and prior weights>
Precompte the "pattern matrix" containing feedbacks for all guess-answer pairs
# Game Loop Begins
while <game not complete>:
    # Computation Phase
    for guess_word in <word list>:
        Pool the total weights from word priors for each coloring pattern via **scatter reduce**
        Compute guess_word's expected entropy I(g) by normalizing the pooled weights into a probability distribution.
        record guess_word's score as a function of its expected entropy. (map reduce)
    # Candidate Selection Phase
    candidate_word = argmax(scores)
    Obtain feedback by submitting candidate_word as guess
    # Solver Update Phase
    for word in <word list>:
        set word's prior weight to 0 if feedback(candidate_word, word) does not match.
    compute the sum of all prior weights to be used in the normalization step
# Game Loop Ends
```

We have also carefully analyzed our wordle solver algorithm, and two of the most important operations to be parallelized in our algorithm are **Map Reduce** and **Scatter Reduce**. The map reduction algorithm is relatively straight forward to parallelise, and many parallel computing frameworks, such as OpenMP and MPI, provides primitives to efficiently perform reductions; it should serve as a reference point for parallelization of scatter reduce. The Scatter Reduce method is characterized by write contention and memory access indirections, both of which limits its parallelism potentials. We have proposed 3 potential approaches to parallel scatter reduce to be experimented with: a lock based approach, a reduction based approach and a reformulation of the scatter reduce problem as a sparse matrix vector multiplication problem. We shall further evaluate the effectiveness of each approach by running each scatter reduce implementation on very large matrices and vectors.

As mentioned in the proposal document, during the computational phase, we could either parallelize over the "guess_word" dimension ("across-word parallelism"), which we believe is the natural approach to parallelize our solver, or experiment with "within-word parallelism" by incorporating our parallel scatter reduce. We are also interested in incorporating SIMD to further enhance the performance of our CPU solver. In the following weeks, we are also interested in completing a CUDA implementation of our wordle solver. Although the pattern matrix pre-computation phase is massively data parallel, there will be challenges in our actual solver loop routine, as CUDA does not have natural reduction primitives and has very limited synchronization constructs. This forces us to reconsider our current scatter reduce and map reduce implementations.

We have also decided modify the overall directive of this project. As mentioned in [_3Blue1Brown_'s addendum to their wordle video](https://www.youtube.com/watch?v=fRed0Xmc2Wg), most of the algorithmic refinemenets included in the version 2 solver are aimed at reducing the average rounds required for a correct guess. However, since we are mostly interested in improving and measuring the parallel performance of the wordle solver on different machine architectures, it is likely that we will sideline the inclusion of (or simplify) some features of _3Blue1Brown_'s algorithm in our parallel implementations. In the "Updated Project Goals and Schedules" section, we have modified our objectives to reflect this change of project directive.

## Deliverable at Poster Session

We plan on showcasing the following items during the poster session:
- Summary of our wordle solver routine and approaches we experimented with in our parallel implementations
- A detailed section describing the differences between a CPU solver and a GPU solver
- Tables displaying the performance characteristics of our CPU and GPU implementations and under different parallel frameworks (OpenMP vs MPI)
- Graphs displaying the scaling of our CPU solver on different core counts
- Graphs displaying the parallel performance of our GPU and CPU solvers on various problem banks.
- \[If we are able to obtain meaningful results from our parallel scatter reduce implemetations \] Showcase the performance characteristics of different implementations of our parallel scatter reduce subroutines. (This is likely to be included in the final report regardless of our success.)


## Preliminary Testing Results

The following preliminary results are obtained on Songyu Han's personal computer. (CPU: 8 Core i9-9880H @ 2.3GHz, hyperthreading enabled) 

This is an evaluation of our across-candidate OpenMP parallel approach. Performance is evaluated using the full wordle word bank (12953 words) and a 10-word test set. 

| Sub Routine (Time: ms) |  Serial  | 2 Threads | 4 Threads | 8 Threads | 16 Threads |
| ---------------------: | :------: | :-------: | :-------: | :-------: | :--------: |
| Program Initialization | 149.27   | 135.41 | 135.30 | 141.46 | 130.43 |
| Coloring Matrix        | 6750.3   | 3530.2 | 1898.1 | 911.63 | 837.75 |
| avg. Time per Game     | 807.04   | 415.74 | 227.25 | 117.72 | 91.62  |

Performance statistics of our MPI implementation as well as variations of our OpenMP approaches will be included in the final report. Additionally, although our implementation scales rather well on this particular machine, we are curious if a GPU implementation would take advantage of the massive data parallel capabilities and higher memory bandwidth.

## Concerns and Unknowns

Our current progress mostly aligns with our initial expectations and it is likely that we will be able to complete all the "plan to achieve" deliverable items. However, due to the delays in the project proposal phase, it is expected for us to perform most of the parallel developmental work in the next two weeks and spend the last week conducting parallel program experimentations.

One item worth mentioning is that we are somewhat underwhelmed by the performance of parallel scatter reduce. On one of the team member's personal computer (Macbook Pro 16 inch, 8 Core i9-9880H CPU), all of our current parallel scatter reduce implementations perform significantly worse than our sequential baseline on the typical input sizes used in the wordle solver (10 to 20 thousand input items, and an output dimension of 243), and the performance gain due to parallelism on very large data sets is also very limited. We propose improving the locality of memory access by sorting the input arrays, but we are especially concerned as sorting inherently requires more work than the actual scatter reduce operation. However, it is likely for us to incorporate some ideas in the sparse matrix vector multiplication approach into our CUDA scatter reduce implementtion.

## Updated Project Goals and Schedules (As of May 6th)
- Week 4/12:
  - 🔴 Provide a serial CPU C++ and pytorch (python with C++ bindings) implementation of the V1 algorithm. (Completed, SH & JY)
  - 🔴 Analyze sequential algorithm and determine multiple parallel appraoches to the Wordle solver. (Completed, SH & JY)
  - 🔴 Implement first OpenMP parallel program on the candidate guess loop. (Completed, SH)
  - 🔴 Provide test-bank evaluator and profile the two implementations on 5-letter wordle. (Completed, SH & JY)
- Week 4/15 (First Half):
  - 🔴 Implement and profile guess level and candidate level parallelism while optimizing for memory locality and minimizing scatter reduce contention. (Finished, SH & JY)
  - 🔴 Evaluate performance characteristics of various implementations of scatter reduce and map reduce. (Finished, JY)
- Week 4/15 (Second Half):
  - ⚫ Evaluate correctness degradation with reduced synchronization on reduction. (Task Removed: Not Necessary for this study.)
  - 🔴 Implement message-passing solver with MPI (Finished, SH)
  - 🔴 Start to implement GPU solver in CUDA (Finished, SH)
- Week 4/22 (First Half):
  - 🔴 Experiment with coloring matrix partitioning or on the fly coloring computation. (Completed, SH)
  - 🔴 Profile and optimize workload balancing across turns. (Completed, JY)
- Week 4/22 (Second Half):
  - 🔴 Continue Optimizing the CUDA Solver (Somewhat Completed, SH)
  - 🔴 Scale problem size in number of letters (up to 7) (Completed, JY)
- Week 4/29 (First Half):
  - 🔴 Profile and analyze performance characteristics of GPU solver (Completed, SH)
  - 🔴 Perform problem size sensitivity analysis (Completed, JY & SH)
  - ⚫ Hope to Achieve: extend problem size in number of boards to solve. (Task Removed: Not likely to finish.)
- Week 4/29 (Second Half): 
  🔴 Writing up report and preparing poster. (Completed, SH & JY)
