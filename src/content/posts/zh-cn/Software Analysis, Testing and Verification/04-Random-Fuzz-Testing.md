---
title: "[SATV 0x04] Random (Fuzz) Testing"
date: 2025-10-10
draft: false
math: true
categories:
  - 软件分析测试与验证
tags:
  - fuzz
---

*Random Testing*, aka *Fuzz Testing*, tests a program by feeding it random inputs and observing whether it behaves "correctly", i.e., whether its execution satisfies a given specification or it simply doesn’t crash. We can consider random testing a special case of mutation analysis.

Fuzzers have gone through three generations. The first generation of Fuzzers just generated random inputs to run the program. This could result in poor coverage.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Random-(Fuzz)-Testing-01.png)

<center>First Generation of Fuzzers</center>

The second generation of Fuzzers adds an input queue to the first generation of Fuzzers. It takes one input at a time from the input queue, mutates it, and gets a series of mutants that are then used as input to the program.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Random-(Fuzz)-Testing-02.png)

<center>Second Generation of Fuzzers</center>

The third generation of Fuzzers builds on the second generation of Fuzzers by deciding whether the current input is worth keeping or not based on feedback from running the program; if it is, it will add the input used for this test back to the input queue, otherwise discard it.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Random-(Fuzz)-Testing-03.png)

<center>Third Generation of Fuzzers</center>

Fuzzers can be classified into three types:

* Mutation-based: introduce small changes to existing inputs that may still keep the input valid yet exercise new behavior
* Grammar-based: provide a specification of the legal inputs to a program for systematic and efficient test generation, in particular for complex input formats
* Search-based: adopt search algorithms to reach some targets more quickly

Fuzzers are effective at finding bugs such as memory errors, undefined behavior, assertion violations, infinite loops, concurrency bugs, and so on.

Pros and cons of random testing:

* Pros:
  * Easy to implement
  * Provably good coverage given enough tests
  * Can work with programs in any format
  * Appealing for finding security vulnerabilities
* Cons:
  * Inefficient test suite
  * Might find bugs that are unimportant
  * Poor coverage[^1]

In this lecture, we will look into 3 classic case studies on random testing.

## 01 UNIX utilities: Univ. of Wisconsin’s Fuzz study

Univ. of Wisconsin’s Fuzz study is the first fuzzing study in history, conducted by Barton Miller at the University of Wisconsin.

* 1990: The research study developed a command-line fuzzer, testing the reliability of UNIX programs by bombarding utilities with random data. The test ended up causing 25-33% of UNIX utility programs to crash (dump state) or hang (loop indefinitely).
* 1995: They expanded to GUI-based programs (X Windows), network protocols, and system library APIs. Owing to their effort, systems got better, by not much. "Even worse is that many of the same bugs that we reported in 1990 are still present in the code releases of 1995."
* Later: They studied command-line and GUI-based Windows and OS X apps

Even in 2020, after more than thirty years, it appears that there is still a place for basic fuzz testing.

Fuzz testing is a silver lining for security bugs. What are security bugs? For example, `gets()` function in C has no parameter limiting input length, so programmer must make assumptions about structure of input, which causes reliability issues and security breaches. Solution: Use `fgets()`, which includes an argument limiting the maximum length of input data.

The following table shows the utility bugs that the team found using fuzz testing on Unix, MacOS, and FreeBSD.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Random-(Fuzz)-Testing-04.png)

## 02 C/C++ Programs: Greybox fuzzing in AFL

In previous experiments, we have learned the basic use of AFL. AFL is the third generation of Fuzzers, a simple yet effective fuzzing tool widely used in industry.

AFL offers greybox fuzzing. Greybox fuzzing guides input generation toward a goal based on lightweight program analysis. The AFL workflow can be divided into three main steps:

1. Randomly generate inputs
2. Get feedback from test executions: What code is covered?
3. Mutate inputs that have covered new code

How do we measure AFL's coverage? What metric should we consider? Here we consider branch (edge) coverage. It should be noted that the "branch" here means shifts between basic blocks instead of conditional branches `if-else`. The rationale is that reaching a code location is not enough to trigger a bug, but state also matters[^2].

To calculate branch coverage, we can add an instrumentation at each branching point:

```c
cur_location = /*COMPILE_TIME_RANDOM*/;  
shared_mem[cur_location ^ prev_location]++;  
prev_location = cur_location >> 1;
```

The first statement identifies the current basic block with a random number. The second statement completes a hash mapping from this branch (A→B) to a memory address that records the number of transfers from A to B. The third statement is simple if we overlook the right shift operation. But why `cur_location >> 1`? Actually, it is to distinguish between A→B and B→A.

Inputs that trigger a new edge in the CFG are considered new behaviours. Alternatively, we can treat inputs that trigger new paths as new behaviours, but this is more expensive to track and may face path explosion problems.

To better assess the quality of an input, we need a way to quantify the extent to which it is a new behaviour. Here we use *edge hit counts*, i.e., how many edges the input hits. Usually, approximate counts based on buckets of increasing size will suffice, since we focus on relevant differences in hit counts.

How does AFL maintain the queue of inputs? Mainly three steps:

* Initially: Seed inputs provided by user
* Once used, keep input if it covers new edges
* Add new inputs by mutating existing input

Mutation is an important step in AFL’s workflow: it performs random transformations of bytes in an existing input.

* Bit flips with varying lengths and stepovers
* Addition and subtraction of small integers
* Insertion of known interesting integers
* Splicing of different inputs

## 03 Mobile apps: Google’s Monkey tool for Android

Monkey is one of the most effective GUI testing tools for Android. It generates gestures as inputs to verify whether the behaviours of apps match our expectations.

Grammar of Monkey Events:

```
test_case := event *
event := action(x,y) | ...
action := DOWN | MOVE | UP
x := 0 | 1 | ... | x_limit
y := 0 | 1 | ... | y_limit
```

Quiz: Give the correct specification of TOUCH and MOTION events in Monkey’s grammar using UP, MOVE, and DOWN statements.

Q1: Give the specification of a TOUCH event at pixel (89,215).

A1: `DOWN(89,215) UP(89,215)`

Q2: Give the specification of a MOTION event from pixel (89,215) to pixel (89,103) to pixel (371,103).

A2: `DOWN(89,215) MOVE(89,103) MOVE(37,103) UP(37,103)`

[^1]:The use of fuzzy testing to try to identify potential problems is a common way of testing software, especially related software like compilers. Take compilers for example, lexical analysis, as an earlier stage in the processing of input (e.g., source code) to a program, has a relatively straightforward function of breaking down the input sequence of characters into individual lexical units (e.g., keywords, identifiers, operators, etc.) that have a specific meaning. Random input can easily cover a wide range of different input situations at this stage, thus providing a more adequate test of the lexical analyzer. However, for the subsequent stages, such as grammatical analysis, which requires the combination of lexical units according to grammatical rules, and semantic analysis, which requires the consideration of more complex semantic logic, etc., it is often difficult for a simple random input to meet the reasonable input format and structure requirements of these subsequent stages, and thus it is not efficient to effectively test these later stages through random input.
[^2]:In general terms, for example, a bug in the function A can be found iff a function B is called before A. In that case, reaching the code location in A does not necessarily trigger the bug.
