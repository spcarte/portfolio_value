# Drift-Diffusion with Brownian Motion
Drift-diffusion with brownian motion is a stochastic process that is used to simulate sample paths (akin to a random walk) for random variables. It is modeled for a single stochastic variable with:

$$
X(t) = {\mu}t + {\sigma}W(t)
$$

Where $X(t)$ is the variable value, $\mu$ is the mean of the distribution that defines the variable (also referred to as drift), $\sigma$ is the standard deviation of the distribution tha defines the variable (also referred to as volatility), and $W(t)$ is standard brownian motion, which has the following properties:

1. $W(0)=0$
2. $W$ has independent increments, meaning that the future values of $W$ are independent of past values of $W$
3. $W$ is likely continuous in $t$
4. The increments of $W$ are normally distributed with mean 0 and variance $dt$

The variable path can also be described with the stochastic differential equation (SDE):

$$
dX(t) = {\mu}dt + {\sigma}dW(t)
$$

Where $dW(t)$ is a series of normally distributed random numbers and $W(t)$ is the cumulative sum of the random numbers. The drift-diffusion model can also be used in the multivariate case with:

$$
X(t) = {\mu}t + BW(t)
$$

and

$$
dX(t) = {\mu}dt + BdW(t)
$$

Where there are $n$ samples, $X(t)$ is a vector with length $n$, $\mu$ is a vector of the drift values for each sample with length $n$, $W(t)$ is a vector of independent brownian motion with the length $n$, and $B$ is any $(n,n)$ matrix that satisfies:

$$
BB^T=\Sigma
$$

Where $\Sigma$ is the covariance between variables and $B$ is typically found via the Cholesky decomposition of $\Sigma$. 

## Geometric Brownian Motion for Modeling Stock Returns
Fractional stock returns can be modeled using the same drift-diffusion model with brownian motion:

$$
\frac{dS(t)}{S(t)} = {\mu}dt +{\sigma}dW(t)
$$

```{note}
This model for stock prices uses the same assumptions that for the Brownian motion. A critical assumption is that the stock price can be modeled as a random (non-deterministic) process.
```

Where $S(t)$ is the stock price. Accordingly, the change in stock price is modeled with:

$$
dS(t) = {\mu}S(t)dt +{\sigma}S(t)dW(t)
$$

Alternatively, the stock value can be defined by the following exponential equation:

$$
S(t) = e^{X(t)}
$$

```{note}
This above equation is for the stock price with drift-diffusion rather than the stock returns.
```

The exponential equation can be inverted to the following:

$$
X(t) = ln\begin{pmatrix}S(t)\end{pmatrix}
$$

This equation can be differentiated with It&ocirc;'s lemma, which states that s stochastic function can be differentiated with:

$$
df = f'\begin{pmatrix}S(t)\end{pmatrix}dS(t) + \frac{1}{2}f''\begin{pmatrix}S(t)\end{pmatrix}\begin{pmatrix}dS(t)\end{pmatrix}^2
$$

Where $f\begin{pmatrix}S(t)\end{pmatrix} = ln\begin{pmatrix}S(t)\end{pmatrix}$, since the stock price is assumed to be a stochastic function. The derivatives are:

$$
f'\begin{pmatrix}S(t)\end{pmatrix} = \frac{1}{S(t)}
$$

$$
f'\begin{pmatrix}S(t)\end{pmatrix} = -\frac{1}{S(t)^2}
$$

These derivatives are substituted into It&ocirc;'s lemma, which generates an equation for the log-returns of a stock:

$$
d\begin{pmatrix}ln\begin{pmatrix}S(t)\end{pmatrix}\end{pmatrix} = \frac{1}{S(t)}dS(t) - \frac{1}{2}\frac{1}{S(t)^2}\begin{pmatrix}dS(t)\end{pmatrix}^2
$$

The drift-diffusion equation for the change in stock price can be inserted into the above equation for the log-returns since it directly defines $dS(t)$. First $\begin{pmatrix}dS(t)\end{pmatrix}^2$ must be computed:

$$
\begin{pmatrix}dS(t)\end{pmatrix}^2 = \begin{pmatrix}{\mu}S(t)dt +{\sigma}S(t)dW(t)\end{pmatrix}^2
$$

Which expands to:

$$
{\mu}^2S(t)^2(dt)^2+2{\mu}{\sigma}S(t)^2dtdW(t)+{\sigma}^2S(t)^2\begin{pmatrix}dW(t)\end{pmatrix}^2
$$

Applying the stochastic (It&ocirc;) calculus rules:

$$
(dt)^2 = 0
$$
$$
dtdW(t) = 0
$$
$$
\begin{pmatrix}dW(t)\end{pmatrix}^2 = dt
$$

$\begin{pmatrix}dS(t)\end{pmatrix}^2$ is:
$$
{\sigma}^2S(t)^2dt
$$

Substituting everything back into the equation for log-returns results in:

$$
d\begin{pmatrix}ln\begin{pmatrix}S(t)\end{pmatrix}\end{pmatrix} = \frac{1}{S(t)}\begin{pmatrix}{\mu}S(t)dt +{\sigma}S(t)dW(t)\end{pmatrix} - \frac{1}{2}\frac{1}{S(t)^2}{\sigma}^2S(t)^2dt
$$

This is simplified through the following steps:
$$
d\begin{pmatrix}ln\begin{pmatrix}S(t)\end{pmatrix}\end{pmatrix} = {\mu}dt +{\sigma}dW(t) - \frac{1}{2}\frac{1}{S(t)^2}{\sigma}^2S(t)^2dt \\
\downarrow \\
d\begin{pmatrix}ln\begin{pmatrix}S(t)\end{pmatrix}\end{pmatrix} = {\mu}dt +{\sigma}dW(t) - \frac{1}{2}{\sigma}^2dt \\
\downarrow \\
d\begin{pmatrix}ln\begin{pmatrix}S(t)\end{pmatrix}\end{pmatrix} = \begin{pmatrix}{\mu} - \frac{1}{2}{\sigma}^2\end{pmatrix}dt + {\sigma}dW(t)
$$

Where the $\mu-0.5\sigma^2$ term is sometimes called the log-drift. This equation for log-returns can be easily integrated to model the log stock price:

$$
ln\begin{pmatrix}S(t)\end{pmatrix} = ln\begin{pmatrix}S(0)\end{pmatrix} + \begin{pmatrix}{\mu} - \frac{1}{2}{\sigma}^2\end{pmatrix}t + {\sigma}W(t)
$$

Exponetiating the log stock price equation results in a simple equation for modeling the stock price:

$$
S(t) = S(0)e^{\begin{pmatrix}{\mu} - \frac{1}{2}{\sigma}^2\end{pmatrix}t + {\sigma}W(t)}
$$

Similarly, the percent change in stock price can be found by exponentiating the equation for log-returns:

$$
d\begin{pmatrix}S(t)\end{pmatrix} = e^{\begin{pmatrix}{\mu} - \frac{1}{2}{\sigma}^2\end{pmatrix}dt + {\sigma}dW(t)}
$$

If the drift ($\mu$) and volatility ($\sigma$) are modeled in the same time increments as the model, $dt=1$, which simplifies the equation for percent returns to:

$$
d\begin{pmatrix}S(t)\end{pmatrix} = e^{{\mu} - \frac{1}{2}{\sigma}^2 + {\sigma}dW(t)}
$$

```{note}
The above equation was developed for log returns. As such, the drift and volatility should be computed from the log-returns.
```

## Adjusting the Model for Multiple Stocks
The geometric brownian motion model for estimating stock returns can be easily adjusted for multiple stocks with the same method that was described above for the linear Brownian motion. There is some rather complicated It&ocirc; calculus to develop the equations. However, it is relatively intuitive to understand how the following changes modify the model for multiple stocks

- The returns become a vector with an entry for each stock
- The log-drift is the same as the univariate case, where it is now a vector with an entry for each stock
- There is a different brownian motion for each stock, where they are organized into a vector
- The brownian motions for each stock are correlated with each other based on the covariance between the stocks

As such, the geometric brownian motion equation is updated to:

$$
\begin{Bmatrix}d\begin{pmatrix}S(t)\end{pmatrix}\end{Bmatrix} = e^{\begin{Bmatrix}{\mu} - \frac{1}{2}{\sigma}^2\end{Bmatrix} + \begin{bmatrix}B\end{bmatrix}\begin{Bmatrix}dW(t)\end{Bmatrix}}
$$

Where the $B$ matrix in this equation is the "square root" of the covariance matrix, as defined by:

$$
\Sigma = BB^T
$$

Where $\Sigma$ is the covariance between the stocks. The are multiple ways to define $B$, but a common method is through the Cholesky decomposition of $\Sigma$. Further, $\sigma^2$ can be estimated directly from the data for a single stock or by taking the diagonal of $\Sigma$. 