# Countable or Uncountable Tangent Universes?
#### Contents
[1. Repository Structure and Contents](#B)

[2. Dependencies](#C)

[3. Brief Theoretical Backgroung](#A)

[4. Bibliography](#D)


##  <a id="A">(3.)</a> Brief Theoretical Background:
### A Quantum Universe is just a Fluid of Classical Universes

Consider a non-relativistic isolated system of $n\in\mathbb{N}$ degrees of freedom $\vec x=(x_1,...,x_n)\in\mathbb{R}^n$, e.g. the system of all the particles in the Universe. According to quantum mechanics, their state in each time $t$ is given by the complex wavefunction $\psi(\vec x,t)$, the dynamics of which in time are given by the Schrödinger Equation, 

$$\tag{1}
i\hbar \frac{\partial \psi(\vec x,t)}{\partial t} = \sum_{k=1}^n \frac{-\hbar^2}{2m_k}\frac{\partial \psi(\vec x,t)}{\partial x_k}+U(\vec x)\psi(\vec x,t),
$$

where $m_k$ is the constant mass of the $k$-th degree of freedom, $\hbar$ is the so-called Planck constant and $U(\vec x)$ is the potential energy describing the interaction between the degrees of freedom (if we considered a *closed* quantum system instead of an *isolated* one, the potential could be made to be time dependent).

By defining the phase and magnitude squared (related to the probability density of finding the system at a configuration $\vec x$) for the wavefunction $\rho(\vec x,t),S(\vec x,t)$, such that in polar form $\psi(\vec x,t)=\rho^{1/2}(\vec x,t)exp(iS(\vec x,t)/\hbar)$, the Schrödinger Equaiton decouples into two real partial differential equations,

$$\tag{2}
\frac{\partial \rho(\vec x,t)}{\partial t}=-\sum_{k=1}^n\frac{\partial}{\partial x_k}\Big[\rho(\vec{x},t)v_k(\vec x,t)\Big]
$$

$$\tag{3}
-\frac{\partial S(\vec x,t)}{\partial t}=\sum_{k=1}^n\frac{1}{2}m_k v_k(\vec x,t)^2+V(\vec x,t)+Q(\vec x,t)
$$

where we defined the fields

$$\tag{4}
v_k(\vec x,t)=\frac{1}{m_k}\frac{\partial S(\vec x,t)}{\partial x_k}
$$

and

$$\tag{5}
Q(\vec x, t)=-\frac{\hbar^2}{4m_k}\Big(\frac{1}{\rho}\sum_{k}\frac{\partial^2\rho}{\partial x_k^2}+\frac{1}{2\rho^2}\sum_{k}(\frac{\partial \rho}{\partial x_k})^2\Big).
$$

If we interpret $v_k$ as the velocity field for the $k$-th degree of freedom, and therefore following classical mechancis, $S$ as the action (its gradient is the momentum), the two partial differential equations $\eqref{2},\eqref{3}$, are nothing but the continuity equation for a compressible fluid of density $\rho$ and the classical mechanics Hamilton-Jacobi equation. The only unusual term regarding classical mechanics is the potential energy term $Q(x,t)$, which we call the Quantum Potential, and is equal to the curvature (Laplacian) of the density plus the magnitude of the gradient of the density, normalized each by the local magnitude of the density. That is, it is higher where the density gets locally "agglomerated" relative to its surrounding and is minimal when the density gets locally flat.

This decomposition is very well known and forms the basis of the Bohmian quantum theory [[1](#1), [2](#2)] and Madelung's quantum hydrodynamics [[3](#3)].

Let us considers the Lagrangian frame of the fluid, where the velocity field guides the ensemble of trajectories $\vec x(\vec \xi,t)$, 

$$\tag{6}
\frac{\partial \vec{x}(\vec \xi, t)}{\partial t} = \vec v(\vec{x}, t)\Big\rvert_{\vec x=\vec x(\vec \xi, t)},
$$

where each trajectory is tagged by $\xi$, representing its positions at some reference time $t_0$, $\vec x(\vec \xi, t_0)=\vec \xi$.

Then, the Hamilton-Jacobi equation simply reduces to Newton's Second law

$$\tag{7}
\frac{\partial^2 x_k(\vec \xi,t)}{\partial t^2} = -\frac{\partial }{\partial x_k} \Big[ U(\vec x,t)+Q(\vec x,t) \Big]\Big\rvert_{\vec x=\vec x(\vec \xi, t)}.
$$

Note that the density $\rho(\vec x,t)$ and the velocity field $\vec v(\vec x,t)$ contain exactly the same information as the wavefunction $\psi(\vec x,t)$ up to an irrelevant global phase. Thus, the time evolution of the quantum system can equivalently be described instead of using the Schrödinger Equation $\eqref{1}$, by using the Newton's Second Law $\eqref{7}$ and the continuity equation $\eqref{2}$. While the curvature of the density of possible trajectories guides the same trajectories through the Newton's second law, the density is guided by the velocity field of these trajectories through the continuity equation. This manifestly shows the quantum many-body problem. Unlike in classical mechanics, it is not possible to numerically evolve a single trajectory alone, since the information over the rest of trajectories is necessary. Thus, increasing one degree of freedom in the system implies the time evolution of an exponentially more trajectories [[4](#4)]. That's the price of a fluid in configuration space.

If one reads literally this decomposition, the time evolution of a quantum Universe can be understood as an uncountable set of classical Universes (a fluid of classical Universes), that repel each other when they get "too close" to each other in configuration space (their density gets "agglomerated"). The natural question following this is: what if in reality there are not an uncountable number of tangent [\*](#f) Universes, but a (perhaps infinte, but) countable number of them? Such that in the limit the interaction between them leads us to the quantum potential for the fluid. This was first publicly argued by Hall et al. in 2014 [[5](#5)].

Even if this was not true, the fact is that, there is a priori no argument against the fact that there exist interaction forces/potentials between $M$ configuration space $n$ dimensional "Universes", such that if we simulate enough of them $M\rightarrow \infty$, we can reconstruct the wavefunction through their density and velocities, with an arbitrary precision, at any time.


#### <a id="f">[*]</a> Why tangent and not, perhaps, parallel?
 Due to the existence and uniqueness theorems of ordinary differential equations, the ensemble of trajectories calculated from the Schrödinger Equation, neveer cross each other in configuration space. Thus, since they interact with each other (repulsively), they are not parallel, but they cannot cross each other, so they are "tangent". 
 
 An ensemble of tangent Universes like these would be compatible with the fact we perceive particles as point-like and have never experienced an actual superposition. The randomness of quantum mechanics would be due to the fact that we cannot know our trajectory's label $\vec \xi$ (for we cannot access all the degrees of freedom of the Universe at the same time), but physics would this way be deterministic at the ontological level. The collapse of the wavefunction and the closed quantum system Schrödinger Equation for any subsystem of the Universe, would then be derived (not postulated) following a similar procedure to [[7](#7)].


## Bibliography
<a id="r1">[1]</a> 
Bohm, David. *"A suggested interpretation of the quantum theory in terms of" hidden" variables. I."* Physical review 85.2 (1952): 166.

<a id="2">[2]</a> 
Oriols, Xavier, and Jordi Mompart, eds. *Applied Bohmian mechanics: From nanoscale systems to cosmology*. CRC Press, 2019.

<a id="3">[3]</a> 
Madelung, Erwin. *"Quantentheorie in hydrodynamischer Form." Zeitschrift fur Physik 40 (1927): 322.

<a id="4">[4]</a> 
Oianguren, Xabier. *"The Quantum Many Body Problem"*, Bachelor’s Thesis (2020) for the Nanoscience and Nanotechnology Degree (UAB).

<a id="5">[5]</a> 
Hall, Michael JW, Dirk-André Deckert, and Howard M. Wiseman. *"Quantum phenomena modeled by interactions between many classical worlds."* Physical Review X 4.4 (2014): 041013.

