---
title: "[SATV 0x05] Metamorphic Testing & Property-based Testing"
date: 2025-10-10
draft: false
math: true
categories:
  - 软件分析测试与验证
tags:
  - 蜕变测试
  - 性质测试
---

So far, we have learned basic concepts in the field of software testing. Two major challenges in software testing:

1. Input Generation
2. Oracle Construction & Checking

In the former section, we have introduced an automatic input generation method - fuzz testing. In this lecture, we will tackle the second challenge.

## 01 Test Oracles

A *test oracle* is a mechanism to decide whether a test output is correct or not[^1]. 

However, it is important to note that in certain instances, the validation of test outputs can be impractical. 

For instance, it is not feasible to provide the correct solution to a shortest path problem when the input graph is substantial and intricate, which renders the verification of a shortest path algorithm intractable. 

The image search engine *Flickr* is another pertinent case study. The question arises as to how one can ascertain whether the engine returns the correct images related to the specified keywords without the need to undertake an exhaustive visual examination of hundreds of thousands of pictures.

Then what can we do? The answer is *Metamorphic Testing*.

## 02 Metamorphic Testing

### 2.1 Introduction

Metamorphic testing offers an approach to both input generation and test result validation.

A central element is a set of metamorphic relations, which are necessary properties of the target function or algorithm in relation to multiple inputs and their expected outputs.

Despite the fact that we are unaware of the solution to the shortest path problem, we can still utilise the metamorphic test to ascertain whether there is an error in the algorithm. It is evident that if the source and destination are reversed, the shortest paths should be in precisely the opposite order. This metamorphic relation, i.e. `|shortestPath(G,s,d)|=|shortestPath(G,d,s)|`, is essential for the testing. If an input can be found such that the algorithm does not satisfy this relation, then the algorithm is proved to have bugs.

Returning to the *Flickr* example, if restrictions are applied to the search (such as limiting the image size to "large"), it would be reasonable to expect that fewer results would be returned than if no restrictions were applied. So we get a metamorphic relation ` If Q2≡Q1 AND size=large then Count(Q2) ≤ Count(Q1)`. If an input can be found such that *Flickr* does not satisfy this relation, then the engine is shown to have bugs.

To summarize, in metamorphic testing, we first construct an original input as a source test case to obtain an output; then we metamorphose the original input to obtain follow-up test cases and their outputs. Finally, we determine whether the follow-up outputs satisfy the metamorphic relation with the original output. The workflow of metamorphic testing can be abstractly represented in the following figure:

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Metamorphic-Testing-03.png)

This process can be mainly split into 4 steps:

1. Identification of metamorphic relations. (manual)
2. Generation/Selection of source test cases. (automated / reuse)
3. Generation of follow-up test cases. (automated)
4. Checking of metamorphic relations. (automated)

Metamorphic testing is a state-of-the-art testing technique that has been adopted extensively in industry and has yielded numerous successful case studies.

### 2.2 Case Study

There is a category of bugs known as *system settings-related bugs*, aka *setting defects*. These bugs are triggered when the conditions for a function to run normally conflict with the system settings on the device in use, which results in the inability to perform the function. In the majority of cases, system setting-related bugs are *non-crashing functional bugs*. Next, we will try to use metamorphic testing to identify system setting-related non-crashing functional bugs in apps.

The initial step in this process is to identify metamorphic relations. Evidently, it is imperative that a robust app ensures that its behaviour is not affected by system settings. So we can conclude the following metamorphic relation:

**<font color="red">MR</font>: If a given system setting is changed and later restored, an app's behaviors (functionalities) should keep consistent.**

For representation convenience, symbols are defined as follows:

* $e$ - a GUI event
* $l$ - a GUI layout
* $e_c$ - changes a given setting
* $e_u$ - restores the setting
* $e.w$ - target widget of GUI event

Assuming that we are testing an app whose initial layout is $l_{1}$, we will perform a sequence of events $e$ on it and observe its behavior under normal settings. The formal representation of the whole process is as follows:
$$
\text{Source test case:}\quad l_{1}\overset{e_1}{\rightarrow}l_{2}\dots\dots l_{i}\overset{e_i}{\rightarrow}l_{i+1}\dots\dots l_{n}\overset{e_n}{\rightarrow}l_{n+1}
$$
To perform metamorphic testing, we construct a follow-up test case by inserting a pair of events $\langle e_{c},e_{u}\rangle$ which changes a setting and then restores it, into the source test case. If successful, we will get the following observation:
$$
\text{Follow-up test case (successful):}\quad l_{1}^{\prime}\overset{e_1}{\rightarrow}l_{2}^{\prime}\underset{\overset{\uparrow\uparrow}{\langle e_{c},e_{u}\rangle}}{\dots\dots} l_{i}^{\prime}\overset{e_i}{\rightarrow}l_{i+1}^{\prime}\dots\dots l_{n}^{\prime}\overset{e_n}{\rightarrow}l_{n+1}^{\prime}
$$
If the follow-up test case performs as well as the source test case, then it passes the test; otherwise, we should see that one of the events $e_i$ after $\langle e_{c},e_{u}\rangle$ being impossible to perform, i.e. $\exists e_i,e_{i}.w\in l_{i}\land e_{i}.w\notin l^{\prime}_{i}$.
$$
\text{Follow-up test case (failed):}\quad l_{1}^{\prime}\overset{e_1}{\rightarrow}l_{2}^{\prime}\underset{\overset{\uparrow\uparrow}{\langle e_{c},e_{u}\rangle}}{\dots\dots} l_{i}^{\prime}\overset{\not{e_i}}{\rightarrow}\text{unable to proceed due to }e_{i}.w\notin l^{\prime}_{i}
$$

### 2.3 Other Notes

* Metamorphic testing requires good knowledge of the problem domain.

* Different metamorphic relations can have different fault-detection capability.

* Metamorphic relations should be diverse so they exercise different parts of the program.

* Two common approaches for the construction of metamorphic relations: *input-driven* vs *output-driven*.

  Input-driven metamorphic testing focuses on systematically modifying the input to the software and observing whether the output adheres to predefined metamorphic relations.

  Output-driven metamorphic testing focuses on analyzing the output results and using them to infer whether the input transformations are correctly handled by the software.

* Metamorphic relations can be combined.

## 03 Property-based Testing

### 3.1 Introduction

Assuming that we are testing a sort algorithm using metamorphic testing and the metamorphic relation we constructed is that the sorting results of a list of numbers and its permutation should be the same, we write the following oracle:

```cpp
void testSortMetamorphic(int [] list)
{
    int[] permutatedList = permutate(list);
    int[] output1 = sort(list);
    int[] output2 = sort(permutatedList);
    assertEquals(output1, output2);
}
```

The question is - are metamorphic oracles enough? Definitely not, because it is entirely conceivable that the sorting results of all permutations may be identical, yet not correctly sorted, escaping the testing process.

With this in mind, we must rigorously check the ordering of the output list to ensure the functionality of the algorithm. See the following code:

```cpp
@given(some.lists(some.integers()))  // random input generator
void testSortProperty(int [] list)
{
    if (len(list) >= 100)  // condition
    	return; 
    int [] sorted_list = sort(list);
    for(int i=0; i<=len(sorted_list)-1; i++)
    	assertTrue(sorted_list[i] <= sorted_list[i+1])  // property
}
```

In this code implementation, when the length of the list is less than 100 (though no reason for doing so, let's just pretend it is necessary), the `testSortProperty` function will iterate through the entire list to check that it is well ordered by an assertion, which is a property that a correct output should carry.

Property-based testing fundamentally depends on being able to write abstract assertions that are effectively specifications. We conclude the basic idea of property-based testing as follows:
$$
\begin{aligned}
&\text{for all }(x,y,z,\dots)\\
&\text{such that }\mathrm{condition}(x,y,z,\dots)\text{ holds}\\
&\mathrm{property}(x,y,z,\dots,\mathrm{f}(x,y,z,\dots))\text{ is true}
\end{aligned}
$$
where x, y, z, ... are input variables, and `f` is a function that we are testing. `condition` is a predicate that returns true if some property is true for x, y, z, ..., while `property` is a test oracle that returns true if the some property holds true between the inputs in x, y, z, ... and the output of the program `f(x, y, z, ...)`.

Property-based testing pattern:

1. Generating random inputs (How to generate valid inputs?)
2. Ensuring the condition holds
3. Checking whether a property holds on those inputs (How to obtain, represent and validate a property?)
4. Shrinking the counter-example input to obtain the minimal one

### 3.2 Case Study

**DMFs** (Data Manipulation Functionalities) are prevalent in mobile apps, which perform the CRUD operations (create, read, update, delete) to handle app-specific data. Ensuring the correctness of these DMFs is fundamentally important for many core app functionalities. The bugs related to DMFs are named as **data manipulation errors** (**DMEs**), are prevalent but difficult to find.

The illustration below shows an example of a DME encountered when the user tries to rename a folder after adding a file in it.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Property-Based-Testing-01.png)

Let's see how to find this bug with property-based testing.

First, we need to specify the properties of DMFs. An app property $\phi$ of some functionality $\mathrm{F}$ can be specified as:
$$
\phi=\langle \mathrm{Pre,E,Post} \rangle
$$

* $\mathrm{E}$ denotes $\mathrm{F}$ in the form of an event trace
* $\mathrm{Pre}$ is the precondition that must hold before execution of $\mathrm{E}$
* $\mathrm{Post}$ is the postcondition defining the effect on the app state after executing $\mathrm{E}$

Explanation of the formula: generate many random inputs (random UI event traces in our context) to check $\phi$. If some input satisfies $\mathrm{Pre}$ but falsifies $\mathrm{Post}$ after executing $\mathrm{E}$, then a DME is found. The process can also be represented as:
$$
\text{Let }s\text{ be some app state, }s\models\mathrm{Pre}\land s^{\prime}=\mathrm{E}(s)\text{, if }s^{\prime}\not\models\mathrm{Post}\text{, then }\phi\text{ is violated.}
$$
What we need to do next is:

* Randomly interleave the relevant DMFs on the shared app data (with other possible random events) to exhibit diverse app states.
* Use an abstract data model to simulate the data update effect of $E$, thus facilitating property checking; and use the consistency between the app data and UI layouts to check the property.

Since now we introduce the abstract data model, which models the behavior of a bug-free app, the checking procedure becomes simple - we just need to check whether $E(s)=R(s)$, where $R(s)$ is the states of the abstract data model after execution of the actions as $E$.

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Property-Based-Testing-02.png)

The following illustration summarizes the workflow of our approach:

![](/images/Software%20Analysis,%20Testing%20and%20Verification/Property-Based-Testing-03.png)

### 3.3 Other Notes

* Strengths of PBT
  * PBT can generate a large volume of tests with minimal human oversight and cost.
  * PBT encourages us to make our assumptions explicit by defining properties.
  * PBT can find things that a human tester does not think of.
* Challenges of PBT
  * PBT's input domain is not systematic, so it is unlikely to test edge cases like boundary values unless explicitly told to, and is unlikely to achieve good coverage of the software unless specifically guided.
  * PBT only tests the properties that it is given, which are (usually) partial properties. While specifying test oracles that are complete is possible, it is uncommon.
  * Side-effects, dependencies, performance, …
* Metamorphic testing (MT) is a special case of Property-based testing (PBT). In metamorphic testing, `condition` is the relation, $p(i,j)$, on the input, and `property` is the relation, $q(o_i,o_j)$, on the outputs, where $o_i=f(i)$, $o_j=f(j)$.
* There are five approaches to specifying the properties for PBT, and different properties have different fault revealing abilities.
  * Validity Testing: Every operation should return valid results.
  * Postconditions: Postconditions relate return values to arguments of a single call.
  * Metamorphic Properties: Related calls return related results.
  * Inductive Testing: Inductive proofs inspire inductive tests.
  * Model-based Properties: Abstract away from details to simplify properties.

[^1]:A test oracle is a mechanism to determine whether the program's output is correct, while ground truth refers to the actual, real-world facts or reference data used to validate correctness. Ground truth can serve as a test oracle (specifically, a golden oracle), but not all test oracles rely on ground truth.