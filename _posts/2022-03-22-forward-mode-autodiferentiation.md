---
title: The magic behind (forward mode) Automatic Differentiation
author: Alvaro
date: 2022-03-18 11:33:00 +0800
categories: [ML, Deep Learning, Maths]
tags: [maths, programming, python]
math: true
mermaid: false
pin: true
# image:
#   path: /commons/devices-mockup.png
#   width: 800
#   height: 500
---

# Demystifying Automatic Differentiation
---

If you recall from highschool you'll probably remember the definition of a derivative:

$$
\begin{equation}
    f'(x) = \lim_{\epsilon \rightarrow 0} \frac{f(x+\epsilon) - f(x)}{\epsilon}
    \label{eq:deriv}
\end{equation}
$$

Which is the same as saying: "what's the slope of the line which crosses $f(x)$ and $f(x + \epsilon)$?":

![derivativeSlope](https://i.imgur.com/H9fAq1e.png)


This $\epsilon$ is really a perturbation, a small amount of "noise" that is added to $x$ in order to calculate the slope.

What the $\lim_{\epsilon \rightarrow 0}$ tell us is that the smallest $\epsilon$ is, the more accurate will be our calculation of $f'(x)$.

Ideally, if we had infinitely precise computers, it wouldn't be a bad idea to use this method for computing derivatives. Unfortunately, this is not the case. We start having (big) problems as long as we keep reducing $\epsilon$.

## How does a computer read numbers?

Why is that? Why we cannot simply reduce $\epsilon$ as far as we want?

> **TL;DR** The number of bits we have for representing numbers is finite, so, inevitably, we incur in **rounding errors** which create great numerical instability.

Computers use the [IEEE 754 standard](https://en.wikipedia.org/wiki/IEEE_754) for representing floating point numbers (i.e. numbers in $\mathbb{R}$). For example, with 32 bits, a number is represented in the following way:

![Floating point representation](https://i.imgur.com/q8fVi0q.png)

This limits out "resolution" to about $10^{-126}$, which is the smallest absolute number (apart from $0$) we can represent.

Simplifying with an example. Imagine we only have bits for representing up to two significative figures. If we, for example had $\epsilon=0.25$ and we wanted to calculate $\frac{\epsilon}{2} = 0.125$, which is not representable with two decimals, we thus would have to round, having $\epsilon=0.13$.

As you see, we would have introduced an error of $0.005$ ($= 2\%$ w.r.t. the true value). It isn't difficult to imagine this same error repeating when we have 32 bits. (Obviously, this error will also happen if we use larger numbers like 64 or 128 bit floats, but later).

Another problem arises when we compute the derivative using the aforementioned formula $\eqref{eq:deriv}$. Typically, in comparison, the numerator quantity ($f(x+\epsilon) - f(x)$) will be much larger than the denominator ($\epsilon$). In a nutshell, this is also bad regarding roundoff errors.

A possible mitigation for these type of errors could be using two numbers at once: one for the *non-epsilon* related parts of the computation and other for the things which involve epsilon. In other words, separating the perturbation-related computations from the rest.

## How math solves our problem: Enter the Dual Numbers

So, two numbers at once... Doesn't that sound like imaginary numbers? ($a+ ib; a,b \in \mathbb{R}$). Quite a bit, they're 2-dimensional numbers which we operate segregated by dimension (i.e. the imaginary part $i$ never interacts with the real part $a$).

As with imaginary numbers, when mathematicians face problems, they tend to construct crazy (but ingenuous nonetheless) settings in which their problems can be easily solved. This is the case of a branch of mathematics called *Smooth Infinitesimal Analysis*, from where the concept of [**Dual Numbers**](https://en.wikipedia.org/wiki/Dual_number) arise.

Dual numbers are an extension to the real numbers and they're similar to the imaginary ones in the sense that both have "2 dimensions" which are independent.

A dual number takes the form of

$$
a + b \epsilon;\hspace{1em} a, b \in \mathbb{R}
$$

with the property that $\epsilon$ is *nilponent*, which in layman's terms means $\epsilon^2 = 0$.

> Although it can seem that Dual and Imaginary numbers are related, this is not the case. Dual numbers are meant to work with calculus involving really, really small numbers. As simple as I can explain it (probably not rigorous): The reason why $\epsilon^2 = 0$ is that if you take the square of a really really small number, it becomes so small that we can basically consider it 0.


### Wait, you were talking about derivatives

Indeed. Dual numbers, being an extension of reals, allow us to compute both the evaluation of a function $f(x)$ and its derivative $f'(x)$ in one run!

But, how? Let's see an example.

Consider a polynomial series:

$$
P(x) = p_0 + p_1 x + p_2 x^2 + p_3 x^3 + ... + p_n x^n
$$

If we plug $x = a + b\epsilon$, see what happens:

$$
P(a+b\epsilon) = p_0 + p_1 (a+b\epsilon) + p_2 (a+b\epsilon)^2 + p_3 (a+b\epsilon)^3 + ... + p_n (a+b\epsilon)^n
$$

Expanding and reordering the terms...

$$
P(a+b\epsilon) = p_0 + p_1 a + p_2 a^2 + p_n a^n \fcolorbox{black}{red}{+} p_1 b\epsilon + 2p_2ab\epsilon + 3 p_3a^2b\epsilon + np_n a^{n-1}b\epsilon
$$

Note how the original $P(x)$ evaluation is on the left of the red $\fcolorbox{black}{red}{+}$ while on the right is $P'(x)b\epsilon$. 🤯

Naturally, this effect extends to Taylor series


> For those of you who don't remember what a Taylor series is. It's a way of approximating functions around a point $p$ using the derivatives of the same function. They look like: $f(x) = f(x) + \frac{f'(a)}{1!}(x - p) + \frac{f^{\prime\prime}(a)}{2!}(x - p)^2  + ... + \frac{f^n(a)}{n!}(x - p)^n$. If you want to learn more about them, I'll refer you to this wonderful [3Blue1Brown video](https://www.youtube.com/watch?v=3d6DsjIBzJ4).


$$
f(a+b\epsilon) = f(a) + f^\prime (a)b\epsilon + \frac{f^{\prime\prime}(a)}{2!}(b^2 \epsilon^2) + \frac{f^{\prime\prime\prime}(a)}{3!}(b^3 \epsilon^3) + ...
$$

The nice thing is that by definition $\epsilon^2 = 0$, so all the terms on the series $>=2$ become 0. Leaving us simply with:

$$
f(a+b\epsilon) = f(a) + f^\prime (a)b\epsilon
$$


*(**Note**: In our case, the value of $b$ doesn't interest us, so from now on, $b=1$)*

This, in turn, makes us able to operate functions and still have the evaluation and derivative parts separated!

If we have two functions $f$ and $g$,

$$
\begin{align*}
    f: f(a) + \epsilon f^\prime(a) \\
    g: g(a) + \epsilon g^\prime(a)
\end{align*}
$$

we can operate them as:

$$
\begin{align*}
f + g  = \big[ f(a) + g(a) \big] + \epsilon \big[ f'(a) + g'(a) \big]\\
f \cdot g  = \big[ f(a) \cdot g(a) \big] + \epsilon \big[ f(a) \cdot g'(a) + g(a) \cdot f'(a) \big]
\end{align*}
$$

{:.image-caption}
(*the above comes from operating the Taylor expansions using the algebra of dual numbers*).


Let's see a real example using this mathematical tool.

Let $f(x) = 3x^2 + x + 1$. We now that it's true derivative is $f'(x) = 6x + 1$.

Applying the dual number method we've just talked about, we have:

$$
\begin{align*}
f(a + \epsilon) &=& 3(a + \epsilon)^2 + (a + \epsilon) + 1 \\
                &=& 3(a^2 + \cancelto{0}{\epsilon^2} + 2a\epsilon) + a + \epsilon + 1 \\
                &=& 3a^2 + a + 1 + 6a\epsilon + \epsilon\\
                &=& \underbrace{(3a^2 + a + 1)}_{f(a) \text{part}}
                        \fcolorbox{black}{red}{+}
                    \underbrace{\epsilon(6a + 1)}_{f'(a) \text{part}}
\end{align*}
$$

Then, we just have to substitute $a$ for the specific value we want and we get both the function evaluation and its derivative at the point $a$!

🤯

## Math is cool, but how can I implement this?

Ok. Let's implement a (*very simple*) automatic differentiation engine using dual numbers. First, we need to define what is a `DualNumber`, using a class.

```python
class DualNumber:
    def __init__(self, real_part, dual_part):
        self.real = real_part
        self.dual = dual_part

    def __repr__(self):
        return f"{self.real} + {self.dual}ϵ"
```

Having defined the `__repr__` method we can see more clearly what is inside of a `DualNumber` object:

```python
DualNumber(2, 3)
```

```
2 + 3ϵ
```

Then, in order to do anything useful,this dual number class must support the arithmetic operations we defined above.

(_In order to keep things simple I will only implement addition. The class would look as follows_)


```python
class DualNumber:
    def __init__(self, real_part, dual_part):
        self.real = real_part
        self.dual = dual_part

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, Number):
            return DualNumber(self.real + other, self.dual)
        else:
            raise TypeError("Cannot add that to a dual number!")

    def __radd__(self, other):
        # This is for when we do things like: 10 + DualNumber()
        return DualNumber(self.real + other, self.dual)

    def __repr__(self):
        return f"{self.real} + {self.dual}ϵ"
```

Let's define a $sin$ function which can accept with `DualNumber`s

```python
import numpy as np

def sin(x):
    if isinstance(x, DualNumber):
        return DualNumber(np.sin(x.real), np.cos(x.real) * x.dual)
    else:
        return np.sin(x)
```

With al the above done, we can evaluate $sin(x)$ and $sin'(x)$ in one single run!

```python
func_evals = []
deriv_evals = []

xticks = np.linspace(-5,5, 100)

for i in xticks:
    x = sin(DualNumber(i, 1))
    func_evals.append(x.real)
    deriv_evals.append(x.dual)

# visualize the function and its derivative
fig, ax = plt.subplots(figsize=(15,7))
ax.plot(xticks, func_evals, label="sin(x)")
ax.plot(xticks, deriv_evals, label="sin'(x)")
ax.legend()
```
![derivative](https://i.imgur.com/Up7yySN.png)

Although it's still necessary to add support for the other operations (`sub`, `mult`, `div`, ...), it's pretty impressive for this few lines, right?
