---
tags: compilers/note
description: "in this chapter, we are going to take a deeper look at the process of the lexical analysis"
---

# 02 Lexical Analysis

As we've discussed in the previous chapter, the goal of lexical analysis is to segment the code into several lexical units. For humans, it is easy to do the job because there are all kinds of visual clues about where the units and the boundary between the two units lie. But apparently, a lexical analyzer doesn't have such a capability. When we input a piece of code like this:

```c
if (i == j)
    z = 0;
else
    z = 1;
```

what the analyzer sees is like this:

```
\tif (i == j)\n\tz = 0;\nelse\n\tz = 1;
```

This is a linear string with all kinds of symbols included. What the lexical analyzer has to work is to place dividers between different units so it can recognize all the lexical units.

## 2.1 Token Class

Of course, an analyzer doesn't solely put dividers between different units, it also has to classify each element of the string according to the role. We call these roles token classes or sometimes just classes. In English, these classes are nouns, verbs and adjectives, and so on. While in programming languages, these token classes are like identifiers, keywords, numbers, or other individual syntaxes...

Token classes correspond to sets of strings. For example, in most programming languages, identifiers are strings of letters or digits, starting with a letter; an integer is a non-empty string of digits; keywords are just sets of reserved words like "if", "else" or "while"... 

So the lexical analyzer classifies program substrings according to roles and then communicates tokens to the parser. What the analyzer gives the parser is a sequence of pairs that are respectively composed of a class and a substring. Each pair is called the "token" and its form should be like this:
$$
\text{token: <class, string>}
$$
For example, if my string is `foo = 42`, after the lexical analysis it will lead to 3 tokens:
$$
\text{<Identifier, "foo">, <Operator, "=">, <Integer, "42">}
$$
To summarize, an implementation must do two things:

1. recognize substrings, which are called lexemes, corresponding to tokens
2. identify the token class of each lexeme

And the output is a series of pairs which are the token class and the lexeme. Each of these pairs is called a token. 

## 2.2 LA Examples

In this section, we will introduce some examples of the lexical analysis.

Before we show the first example, let's learn some details about Fortran. In Fortran, the whitespace is considered nothing so it will be ignored by compilers, which means `DO 5 I` is equal to `DO5I` in Fortran.Now let's take a look at the following statements in Fortran:

```fortran
DO 5 I=1,25
DO 5 I=1.25
```

As you can see, the two statements above look completely the same except that the comma in the first statement is replaced by a dot in the second statement, which makes all the difference.

The first statement is a do-loop whose label is 5. The variable `I` will go from 1 to 25 and the divider between 1 and 25 is a comma. While the second statement is simply an assignment for the dot between 1 and 25 does not correspond to do-loop's syntax. The substring `DO` is not a keyword here but a part of the identifier of the variable `DO5I` as we've discussed before. So what it does is just assign the value `1.25` to the variable `DO5I`, totally different from the first.

So in this example, in order to figure out the role of `DO`, it is required to look ahead to see whether it is a comma or not. Only if we reach that point can we eliminate the ambiguity. As you can imagine, having a lot to look ahead complicates the process of the lexical analysis. Therefore, one of the goals of designing a lexical system is to minimize the amount of "lookahead" that is required. 

The reason why we put so mucn emphasis on "lookahead" is well explained in the next example:

```python
if i == 2:
	z = 1;
else:
	z = 2;
```

When the analyzer reads the symbol `=`, it has to decide whether it's an assignment operator or a relation operator. Also, when it comes across the character `e`, it has to decide whether it's a variable name or the keyword `else`.

In another early programming language Pascal, keywords are not reserved, which means you can use keywords as variable names. Here is a piece of code in Pascal:

```pascal
IF ELSE THEN THEN = ELSE; ELSE ELSE = THEN
```

This interesting piece of code is a bit annoying for even us humans to understand. If looking closely, you will find that the first word `IF` is a keyword, the second `ELSE` should be an identifier, whose type may be boolean, the third `THEN` ought to be a keyword, the fourth `THEM` is an identifier, and the value of another variable `ELSE` is assigned to it, the sixth `ELSE` is a keyword, and the seventh `ELSE` and the eighth `THEN` are both identifiers. So in Pascal, the lexical analysis is actually a huge challenge.

```c++
Foo<Bar<X>>

cin >> var;
```

Seeing the above code, I believe you have reckoned what problem we are going to look into next, which is a classical problem in C++. It is clear to see that the nested template conflicts with the stream operator. It takes a long for many developers to solve the problem and turns out the perfect solution is to insert a blank into the nested template to separate the greater-than signs. As you can see, it is a kind ugly way to solve the ambiguity in lexical analysis.

These examples make us better understand what the lexical analysis is trying to do:

1. The goal is to partition the string. This is implemented by reading left-to-right, recognizing one token at a time.
2. "Lookahead" may be required to decide where one token ends and the next token begins.

## 2.3 Regular Languages

To briefly review, the lexical structure is a set of token classes and each one of the token classes consist of some sets of strings. We need a way to specify which set of strings belong to each token class and an usual tool for doing that is to use regular languages.

### 2.3.1 Definitions

To define regular languages, we generally use something we call regular expressions. Each regular expression denotes a set. There are two basic regular expressions -- **single character** and **epsilon**:

1. A regular expression containing just a single character like 'c' denotes a language that contains only 'c'. 

2. Another basic component of regular language is the expression epsilon $\epsilon$, which contains only an empty string.

Besides the 2 basic expression, there are 3 compound regular expressions:

1. **Union**
   $$
   A+B=\left\{a\ |\ a\in A\right\}\cup\left\{b\ |\ b\in B\right\}
   $$

2. **Concatenation**
   $$
   AB=\left\{ab\ |\ a\in A \wedge b\in B\right\}
   $$

3. **Iteration**
   $$
   A^*=\bigcup_{i≥0} A^i
   $$
   
   And $A^i$ means that $A$ concates iteself $i$ times. Particularly, $A^0$ is euqal to an empty string, namely the epsilon $\epsilon$.

### 2.3.2 Examples

Here we take some regular expressions that are only compsed by the characters "1" and "0" for examples.

1. $1^*$

   According to the difinition of iteration, $1^*$ is euqal to $\bigcup_{i≥0}1^i$, namely all strings consisting of only 1, including the $\epsilon$. So the answer is $\{\text{"", "1", "11",... ,"11...1",...}\}$.

2. $(1+0)1$

   The answer is $\{ab\ |\ a\in (1+0)\cap b\in 1\}$, and as you may find, the union of 1 and 0 concatenates 1 is euqal to 1 respectively concatenating 1 and 0, whose result is the union of 11 and 01(mind that 01 is not 10).

3. $0^*+1^*$

   The answer is $\{0^i\ |\ i≥0\}\cup\{1^i\ |\ i≥0\}$.

4. $(0+1)^*$

   So at first sight, we know that it is euqal to $\bigcup_{i≥0}(0+1)^i$ but what does the string really looks like? According to the iteration's definition, it can be written as $(0+1)(0+1)...(0+1)$, which is all strings of 0's and 1's. It means if we have a string whose length is $i$, in every position, we can pick a "0" or "1" to plug in.

## 2.4 Formal Languages

**Definition**: Let $\Sigma$ be a set of characters (an alphabet). A language over $\Sigma$ is a set of strings of characters drawn from $\Sigma$. This language is called a **formal language**.

To be more concrete, the alphabate can be compared to English characters and the language is like English sentences. A formal language is a bunch of sentences composed by certain characters under a rule. For example, a formal language could be like:

* Alphabet = ASCII
* Language = C Programs

This is definitely a very well-defined language, which is exactly a set of inputs that a C compiler will accept.

Another important concept in formal languages is **meaning functions**. A meaning function maps syntax to semantics.

Let  *L* be a meaning function, *e* be a regular expression, *M* the corresponding alphabet. We can get a map as below:
$$
L(e)=M
$$
Take an expression from the last section for example, there will be a map like: $L(0^*+1^*)=\{0^i\ |\ i≥0\}\cup\{1^i\ |\ i≥0\}$.

