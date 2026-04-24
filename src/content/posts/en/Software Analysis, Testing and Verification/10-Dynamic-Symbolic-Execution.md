---
title: "[SATV 0x0a] Dynamic Symbolic Execution"
date: 2025-10-10
draft: false
math: true
categories:
  - "Software Analysis, Testing and Verification"
tags:
  - symbolic execution
  - static analysis
---

This lesson introduces dynamic symbolic execution, a powerful automated testing technique that combines static and dynamic analysis to maximize program path coverage and uncover potential bugs. It systematically generates inputs to drive program execution along different paths, is versatile across programming languages, and avoids false positives. Although it may result in false negatives (i.e., it may miss some bugs), its success has led to both open-source and commercial implementations. The lesson will cover its principles and teach you how to apply it to test both small code units and large, complex programs.

<!--more-->

## 01 Introduction

### 1.1 Motivation and Approach

 Writing and maintaining tests is tedious and error-prone. A compelling idea to overcome this problem is automated test generation. This idea has several benefits:

* Generate regression test[^1] suite
* Execute all reachable statements and thereby attain high code coverage
* Catch any assertion violations

Dynamic symbolic execution is an automated test generation technique. This technique keeps track of the program state, both concretely like a dynamic analysis and symbolically like a static analysis. It solves constraints to guide the program's execution at branch points. In this manner, it systematically explores all execution paths of the unit being tested. 

Dynamic symbolic execution is an example of a hybrid analysis: it collaboratively combines dynamic and static analysis.

### 1.2 Computation Tree

A program can be seen as a binary tree with possibly infinite depth called **Computation Tree**. In a computation tree, each node represents the execution of a *conditional statement*; each edge represents the execution of a sequence of non-conditional statements; each path represents an equivalence class of inputs. If two inputs lead to the same set of branch points and statements executed, we consider those inputs to be equivalent.

Loops can also be thought of as a nested branching structure. To include loops in a computation tree, we unroll them by representing each loop as a sequence of consecutive if-then-else statements. This means that our tree might have infinite depth, as some loops may be unbounded.

Let's see an example of computation tree.

```c
void test_me(int x, int y) {
    if (2 * y == x) {
        if (x <= y + 10)
            printf("OK");
        else {
            printf("something bad");
            ERROR;
        }
    } else
        printf("OK");
}
```

The program takes as input two integer variables `x` and `y`. It first tests whether `2*y==x`. If `2*y!=x`, then the program exits normally. But if `2*y==x`, then the program proceeds to test whether `x<=y+10`. If `x<=y+10`, then the program exits normally. But if `x>y+10`, then the program throws an error. Thus, we can draw its computation tree like:

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Dynamic-Symbolic-Execution-01.png)

Because the program has no unbounded loops, this computation tree is finite.

### 1.3 Automated Test Generation Approaches

To better motivate the dynamic symbolic execution approach, let's look at some existing approaches for automated test generation.

First, we will consider random testing, in which we generate random inputs and execute the program on these generated inputs.

However, random testing is not always effective. See the following code:

```c
void test_me(int x) {
    if (x == 94389) {
        ERROR;
    }
}
```

The program `test_me` takes an integer `x` and if `x` equals `94389`, it raises an error. Otherwise, the program exits normally. Assuming an int is 32 bits and each possible int has an equal chance of being generated, the probability that our random input will detect this error is astronomically small: one out of 2 to the 32nd power, which is about 23 billionths of a percent. So there is an extremely high possibility that random testing will generate a false negative in a limited amount of time.

---

Another approach that has existed since the 1970s is called "**Symbolic Execution**." In this approach, input variables are represented symbolically instead of by concrete values. The program is executed symbolically, and symbolic path constraints are collected as the program runs. At each branch point, we invoke a **theorem prover** to determine whether a branch can be taken; if so, then we take the branch, otherwise, we ignore the branch as dead code.

See the following code:

```c
void test_me(int x){
    if (x * 3 == 15) {
    	if (x % 5 == 0)
    		print("ok");
        else {
            print("something bad");
            ERROR;
        }
    } else
    	print("ok");
}
```

For example, in this new version of the program `test_me`, instead of testing that the input variable `x` equals a particular integer value, we ask the theorem prover if there is any integer value for `x` that satisfies the condition `x*3!=15`. The theorem prover would respond "yes", allowing us to deduce that the "false" branch is reachable. Because the false branch terminates the program, we now ask the theorem prover if the negation of `x*3!=15` has any satisfying assignment (that is, does `x*3==15` have a satisfying assignment?). The theorem prover would respond "yes," so we'd explore the "true" branch of the first condition while "collecting" the symbolic constraint that `x*3==15`.

Next, we reach the second condition `x%5==0`. We now ask the theorem prover if the expression `x*3==15 AND x%5==0` has a satisfying assignment. The theorem prover would respond "yes," so we'd explore the "true" branch, leading to program termination. Finally, we negate the condition and ask if `x*3==15 AND x%5!=0` has a satisfying assignment. The theorem prover would respond "no," meaning that the false branch is unreachable, dead code. We therefore skip that branch. Since we have then explored all feasible paths and not reached an error, we can conclude that the program will not raise an error in any execution.

However, because of the possibility of exponential explosion in branch conditions, it quickly becomes obvious that this strategy does not scale to large programs.

Now, let's see an example where symbolic execution fails:

```c
void test_me(int x) {
    if (pow(2,x) % c == 17) {
        print("something wrong");
        ERROR;
    } else
        print("OK");
}
```

In this example, the theorem prover needs to decide whether there exists an integer `x` such that 2 to the power `x` modulo a product of two large prime numbers, denoted here by constraint `c`, equals 17. However, this particular condition is an instance of the discrete logarithm problem, which is believed to be computationally intractable on classical computers. In this case, the symbolic execution approach would yield a false positive, considering both branches of the condition to be reachable when really only the "false" branch is reachable.

## 02 The DSE Approach

### 2.1 Introduction

In this course, we are going to combine random testing and symbolic execution together to get the benefits of both without suffering from the limitation of either. This approach is called **Dynamic Symbolic Execution** (**DSE**).

Here's how it works:

1. It initially sets the input values of the function to be tested randomly, and observes the branches of computation that are taken. It also keeps track of constraints on the program's state symbolically.
2. Upon reaching the end of some computational path, DSE will backtrack to some branch point and decide whether there is a satisfying assignment to the program's input variables that allows the other branch to be taken at that point. If so, the solver generates such an assignment, and DSE continues onward. If not, then DSE ignores that branch as dead code.
3. If a condition becomes complex enough that the solver cannot find a satisfying assignment, then the solver "plugs in" the concrete values that DSE is working with to one or more variables in the constraints to simplify them. This strategy makes the constraint solver into what is called an "incomplete" theorem prover: it will never declare an unsatisfiable constraint to be satisfiable, but it may fail to satisfy some satisfiable constraints because of the simplification being made. (This contrasts with pure symbolic execution, whose constraint solver was unsound -- it would declare some unsatisfiable constraints to be satisfiable.)

### 2.2 Example 1

Let's walk through an example of how DSE would identify failure-generating inputs to a function. In this example, we will be looking at the following functions: `foo`, which takes an int `v` and returns the int 2 times `v`; and the function `test_me`, which takes two ints, `x` and `y`, and has no return type. `test_me` operates as follows: it sets the int `z` equal to `foo` of `y`. Then, if `z` equals `x`, it makes an additional check if `x` is greater than `y` plus 10. If this second check passes, then the program throws an error. If either of the checks fail, then the program terminates without error.

The process is too hard to describe in plain text, so I made a gif to show it. Note that the initial values of `x` and `y` are randomly picked, and every time DSE tries to take the "true" branch, it will ask a solver to find a satisfying assignment that can satisfy the condition.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Dynamic-Symbolic-Execution-02.gif)

### 2.3 Example 2

The above example is simple, so the solver can easily find satisfying assignment. Now let's see a more complex one where the solver is not able to do that.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Dynamic-Symbolic-Execution-03.gif)

See? In this case, dynamic symbolic execution, unlike symbolic execution, uses its concrete state to simplify the symbolic constraint. Here, it replaces `y0` in the symbolic constraint by 7, the concrete value of `y` in the program at that point. Since the symbolic constraint here is `secure_hash(y0)==x0`, we obtain a new value of `x0` that is 60...129. Problem solved!

### 2.4 Example 3

Does DSE always work? Not really. Next, let's see an example where DSE fails.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Dynamic-Symbolic-Execution-04.gif)

In order to take the "true" branch of the second condition, we need to find a satisfying assignment to the path condition with the most recently added constraint negated: that is, we need to find `x0` and `y0` with the same `secure_hash` but so that `x0!=y0`. Finding such a pair of inputs -- called a collision -- is a hard problem for cryptographically secure hashes, so our solver is likely not going to be able to find them.

As we can see, the solver tries to replace `y0` by 7 and `x0` by 22, but finally the condition is not satisfiable. This means that DSE would not find the error in the code, as the branch it lies on is considered to be unreachable. In this example, DSE has returned a false negative: it has failed to find the error in the code.

### 2.5 Example 4

So far we have seen DSE's usefulness in the context of testing functions as units. Now let's take a look at how DSE could be used when the unit of test is a data structure.

```c
typedef struct cell {
    int data;
    struct cell *next;
}
int foo(int v) { return 2*v + 1; }
void test_me(int x, cell *p) {
    if (x > 0)
        if (p != NULL)
            if (foo(x) == p->data)
                if (p->next == p)
                	ERROR;
    return 0;
}
```

Here is what a typical random test driver would do. It would generate a random value of `x` and a random memory graph (filling in random values for the data field in each node) reachable from an initial pointer `p` to give to `test_me`. The probability that even the third condition, `foo(x)==p->data`, is true is extremely small (in fact, 0 if `p->data` is even). So it's highly unlikely that the error in this function would be caught by a random tester.

Dynamic symbolic execution, on the other hand, would find this error after at most five runs of the `test_me` function. Here is how it does that:

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Dynamic-Symbolic-Execution-05.gif)

### 2.6 Properties of DSE

The difference between dynamic symbolic execution and "pure" symbolic execution is therefore similar to the difference between dynamic and static analysis. Dynamic analysis will never model a run of the code that could not actually occur, so it will never return false positives: in other words, dynamic analysis is complete. But it can miss actual runs of the code that lead to errors, so it is not sound.

In contrast, symbolic execution on its own will always take a branch that it isn't sure cannot be reached. So it may model runs of the program that could never happen, sometimes returning spurious errors (hence it is incomplete), but it will take all reachable branches as well, so it will never incorrectly declare a program to be error-free (hence it is sound).

Is DSE guaranteed to terminate? In a general program with unbounded loops, DSE is not guaranteed to terminate. For example,

```c
void test_me(int x) {
    while (x > 0)
        x = x - 1; // infinite loop
}
```

However, note that we could make DSE always terminate by specifying an arbitrary stopping condition (e.g., go no deeper than 50 branch points).

Is DSE complete? Is DSE sound? It is complete but not sound, as discussed above.

In the landscape of testing techniques, there are two separate spectra: automated versus manual and black-box versus white-box. The quadrant of the landscape in which DSE falls is automated & white-box. It is an algorithmic technique for deriving inputs leading to programming errors, so it is certainly an automated technique. Moreover, it requires access to the program's code, so it is unequivocally a white-box technique.

Different automated tools have different strategies for searching the space of inputs for error-producing inputs. DSE is an example of a systematic search: even though its first input is randomly generated, all remaining inputs are derived by systematically solving constraints relevant to the program's computation tree.

By its very nature, DSE is a path-sensitive static analysis. The basis of its operation requires distinguishing between different paths in a program's computation tree.

Finally, the instrumentation in DSE is non-sampled. Non-sampled instrumentation means that every relevant instruction or control flow point in the program is instrumented (i.e., modified to collect runtime information).

### 2.7 Approach in a Nutshell

Dynamic symbolic execution is a hybrid approach to software testing that attempts to strike a balance between the costs and benefits of dynamic and static analysis. As you saw, it generates concrete inputs one-by-one such that each input takes a different path through the program’s computation tree. And it executes the program both concretely and symbolically.

These two types of execution cooperate with each other. On the one hand, the concrete execution guides the symbolic execution. By replacing symbolic expressions with concrete values if the symbolic expressions become too complex, the concrete execution enables DSE to overcome the incompleteness of the theorem prover.

On the other hand, the symbolic execution allows DSE to generate new concrete inputs for the next execution of the program. This increases the coverage potential of DSE over other dynamic analyses such as pure random testing.

[^1]:Regression testing aims to verify that the software has not been modified or updated in a way that introduces new problems or breaks functionality that was previously working properly.