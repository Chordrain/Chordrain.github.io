---
tags: compilers/note
description: "this chapter is just an introduction to compilers that offers an overview of the course"
---

# 01 Intro to Compilers

## 1.1 Compilers vs Interpreters

There are two major approaches to implementing programming languages -- compilers and interpreters. This class is mainly about compilers but it's still necessary for us to take a quick look at the differences between them.

What interpreters do is take the programs we wrote and whatever data we want as input and produce the output directly, which means they don't need any preprocess before executing the program and getting the output. We just write the program and simply involve it with our data and the program immediately begins running. In this case, we can say that the interpreter works **online**.

However, compilers structure differently. Compilers take as input just your program and then produce an executable. The executable is another program, maybe assembly language, byte codes, or any other implementation languages, and can be run separately on your data and produce the output. In this structure, the compiler is **offline**. The process of compilation is essentially a preprocessing step that produces an executable program.

## 1.2 History of Compilers

The story begins in the 1950s, particularly with a machine called 704 built by IBM. A interesting thing that the customers who bought the machine and were using it found is that, the software costs exceeded the hardware costs, not just a little bit but a lot. At that time, the hardware was already extremely expensive, but even then, the software was the dominant expense in making good use of computers. This lets a number of people think about how they could do a better job writing software and making programming more productive.

The earliest attempt to improve the productivity of programming is SpeedCoding, the first high-level language for IBM computers, developed in 1953 by John Backus. SpeedCoding is what we will call today an early example of interpreters. The primary advantage of it is that it's much faster to develop a program, but on the other hand, it's 10 to 20 times slower to run the program. Also, it takes up about 300 bytes of memory, which was 30% of the memory on the machine back then. 

In the end, SpeedCoding did not go popular but it gave John Backus the idea of another project. He believed that the main problem of SpeedCoding is that it used the interpreter to parse formulas and if they could interpret the formulas into a form that the machine could execute directly, the code would be faster. While still allowing the programmers to use the language at a high level, thus was the FORTRAN project formed, which is short for "Formulas Translated". (One funny fact about it is that the developers thought they could finish the development in one year but it finally took them 3 years to finish developing in 1957.)

So FORTRAN Ⅰ is the first successful high-level language and exerted a huge impact on computer science. In particular, it led to an enormous body of theoretical work. The impact of FORTRAN Ⅰ is not only just on programming languages but also on the development of compilers. Many modern compilers preserve the outline of FORTRAN Ⅰ.

## 1.3 Structure of FORTRAN Ⅰ

The structure of FORTRAN Ⅰ includes 5 phases:

1. **Lexical Analysis**
2. **Parsing**
3. **Semantic Analysis**
4. **Optimization**
5. **Code Generation**

Lexical analysis and parsing together take care of the syntactic aspect of language. Semantic analysis of course takes care of the semantic aspect like types and scope rules. Optimization is a translation of the program to either let it run faster or use less memory. And the last code generation does a translation to another language and depending on our goal, it might be machine codes or byte codes for a virtual machine or might be another high-level programming language.

## 1.4 Structure of a Compiler

In the last section, we've introduced the structure of FORTRAN Ⅰ, which is also the structure of a compiler. In this section, we're going to have a deeper insight into the 5 phases.

### 1.4.1 Lexical Analysis

The lexical analysis aims to divide program text into "words" or "tokens". As a human, to understand a sentence, we first need to understand the words. Depending on the blanks between each word, we can tell the words that constitute the sentence. So does the machine. Now take a look at the following statement:
$$
\text{if x == y then z = 1; else z = 2;}
$$
A machine also needs to recognize the words and tokens in a statement so that it can understand the meaning of it. Just as the statement shown above, it consists of English words, operational characters, punctuations, and blanks. The lexical analysis aims to differentiate those words and tokens, solving the problems like why the double equal sign is not simply two equal signs connecting.

### 1.4.2 Parsing

Once the words are understood, the next step is to understand the sentence structure. Still taking the former statement for example, we can tell that it is a if-then-else statement so the root of the diagram of our parse tree is going to be if-then-else.

If-then-else consists of 3 parts, namely a predicate, a then-statement, and an else-statement. Let's first look at the predicate:
$$
\text{x == y}
$$
The predicate includes 3 pieces -- a variable x, a comparable operator, and another variable y. Together those form a relation[^1] and through the comparison between the two things, you can have a valid predicate.
$$
\text{z = 1; z = 2}
$$
Similarly, as what's shown above, the then-statement encompasses an assignment, and the else-statement also has one.

All of these form the if-then-else's parse tree and down below is its structure:

![](笔记库/专业课/Compilers/attachment/1.png)

### 1.4.3 Semantic Analysis

Once sentence structure is understood, we can try to understand "meaning". Unfortunately, we don't know how the human brain functions to understand the meaning of a sentence, nor do we know what is going to happen after lexical analysis and parsing, which makes semantic analysis the hardest part. So what we need to understand is, in terms of semantic analysis, compilers can solely perform limited semantic analysis to catch inconsistencies, which means if the program is somehow self-inconsistent, compilers can often notice that and report those errors but they don't know what the program is intended to do.

Let's observe the following English sentences:
$$
\begin{align}
\text{Example:}&\\
&\text{Jack said Jerry left his assignment at home.}\\
\text{Even worse:}&\\
&\text{Jack said Jack left his assignment at home.}
\end{align}
$$
The first sentence is ambiguous for we can't ascertain which person the "his" is referring to. While the second one is even worse considering that we don't even know how many people indeed exist in the context. There can be up to 3 people if the two "Jack"s and the "his" respectively refer to three different people, or maybe there is only 1 person named Jack, and the 3 personal pronouns all point to the same Jack.

This kind of ambiguity is a real problem in programming languages, which is called variable bindings. So to solve that problem, programming languages define strict rules to avoid ambiguities. Think about the codes below:

```c
{
    int Jack = 3;
    {
        int Jack = 4;
        cout << Jack;
    }
}
```

It is easy to solve the output to be 4 because we all know that the outer definition "Jack" is to be hidden by the inner definition if you have learned C or other alike languages. That's just a standard rule of a lot of lexically scoped programming languages for solving ambiguities. But still, it's too hard for compilers to understand a sentence.

### 1.4.4 Optimization

Optimization has no strong counterpart in English but it's a little bit like editing, which automatically modify programs so that they can run faster and use less memory.

Here is a simple example of the kind of operations the optimization program might do:
$$
\text{X = Y * 0 is the same as X = 0}
$$
Anyone who has received primary eduaction knows the two equations above are totally equal. So if our compilers can learn to transform the former one into the latter, it will save some computations and run faster. But unfortunately, this is not a correct rule. That's because this rule is only valid for integers, not for all data types like floating points. Why is it? If you know the IEEE standard well, you will find there is a special number called "Not a Number"(nan). Particularly, nan multiplies any number would produce a nan instead of a zero. So clearly compilers can't do the optimization for it will break an important algorithm underlying the proper propagation of not numbers.

### 1.4.5 Code Generation

Code generation produces assembly code in most cases but generally, it is just a translation into another language, analogous to human translation. Just as humans translate English into Chinese, compilers translate a high-level language into assembly code.

### 1.4.6 Summary

The overall structure of almost every compiler adheres to our outline but the proportions have changed since FORTRAN. The early compiler of FORTRAN Ⅰ has rather sophisticated phases of lexical analysis, parsing, optimization, and code generation and a simple phase of semantic analysis. Different from the early version, today's compilers tend to have small phases of lexical analysis, parsing, and code generation, a more involved phase of semantical analysis, and an extremely complicated phase of optimization which has become the dominant component of all modern compilers.

![[笔记库/专业课/Compilers/attachment/2.png]]

[^1]:关系表达式