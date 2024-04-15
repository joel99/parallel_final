# Project Checkpoint Report
Final project for 15418/618. Songyu Han and Joel Ye

[Project Page](https://joel99.github.io/parallel_final/)



## Work has done so far
- What we have learned
- What we have implemented
- Different implementational decisions
- Different levels of parallelism
- Discuss 
- 


## List of Concerns

## Updated Project Schedules
- 4/12:
  🔴 Provide a serial CPU C++ and pytorch (python with C++ bindings) implementation of the V1 algorithm. (Completed)
  🔵 Add naive OpenMP parallelism on guess and candidate loops. (In Progress)
  🔴 Provide test-bank evaluator and profile the two implementations on 5-letter wordle. (Completed)
- 4/15 (4/16 milestone report)
  🔵 Implement and profile guess level and candidate level parallelism while optimizing for memory locality and minimizing scatter reduce contention. (In Progress)
  ⚫ Evaluate correctness degradation with reduced synchronization on reduction. (Pending)
  🔵 Evaluate performance characteristics with multiple implementations of scatter reduce and map reduce (In Progress)
  ⚫ Implement message-passing solver with MPI (Pending)
- 4/22
  ⚫ Experiment with coloring matrix partitioning or on the fly coloring computation (all in OpenMP). (Pending)
  ⚫ Profile and optimize workload balancing across turns. (Pending)
  🔵 Scale problem size in number of letters (up to 7) (In Progress)
  ⚫ Implement and optimize GPU implementation of the wordle solver in CUDA (Pending)
- 4/29
  ⚫ Hope to Achieve: extend problem size in number of boards to solve. (Pending)
  ⚫ Perform problem size sensitivity analysis (Pending)
- 5/5  
  ⚫ Writing up report and preparing poster. (Pending)
