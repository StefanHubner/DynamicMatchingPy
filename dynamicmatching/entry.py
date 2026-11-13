import jax
import jax.numpy as jnp
import numpy as np


def entry_mass(mus, logits):
    return mus[:, 0, 0:1] * jax.nn.sigmoid(logits)


def empirical_entry(theta, transitions, incumbent):
    p0, q0, p1, q1 = transitions[2:]
    return (lambda mus, e, d: incumbent(mus, d, p0, q0, p1, q1),
            lambda mus, logits: jnp.zeros(mus.shape[0], dtype=mus.dtype),
            lambda logits, gain: jnp.zeros_like(logits),
            lambda mus, logits: (0.0, np.empty((0, 2))))


def empirical_spec(data):
    return empirical_entry


def entry_fit(data):
    years, pop = np.asarray(data['years']), np.asarray(data['population'])
    periods = ((years >= 2001) & (years <= 2008), years >= 2009)
    eligible = np.asarray(data['eligible'])[1:].mean(axis=1)
    observed = np.asarray(data['entered'])[1:].mean(axis=1) / eligible
    # Regime totals are treated as representative of these fitted years.
    weights = eligible / np.array([pop[i].mean() for i in periods])

    def fit(mus, logits):
        n = np.asarray(mus, dtype=np.float64)[:len(years), 0, 0]
        ell = np.asarray(logits, dtype=np.float64)[:len(years), 0]
        scores, rates = [], []
        for i, observed_p, weight in zip(periods, observed, weights):
            w = pop[i] * n[i]
            logw = np.log(w)
            logp = np.logaddexp.reduce(logw - np.logaddexp(0, -ell[i])) - np.log(w.sum())
            logq = np.logaddexp.reduce(logw - np.logaddexp(0, ell[i])) - np.log(w.sum())
            scores.append(weight * (observed_p * (np.log(observed_p) - logp)
                          + (1 - observed_p) * (np.log1p(-observed_p) - logq)))
            rates.append((observed_p, np.exp(logp)))
        return np.sum(scores, dtype=np.float64), np.asarray(rates)
    return fit


def endogenous_entry(data):
    fit = entry_fit(data['entry'])

    def build(theta, transitions, incumbent):
        kernels = transitions[2:]
        stay = jnp.eye(kernels[0].shape[0], dtype=theta.dtype)[0]
        roles = [k[0, 0].at[0].set(0) for k in kernels]
        delta = jnp.stack([jnp.concatenate([r / r.sum() - stay for r in roles[i:i+2]])
                           for i in (0, 2)])
        kernels = tuple(k.at[0, 0].set(stay) for k in kernels)
        p0, q0, p1, q1 = kernels
        kappa, s = theta[-2], jnp.exp(theta[-1])

        def population(mus, e, d):
            change = delta[0] + d * (delta[1] - delta[0])
            return incumbent(mus, d, p0, q0, p1, q1) + e * change

        def payoff(mus, logits):
            p = jax.nn.sigmoid(logits)
            entropy = -p * jax.nn.log_sigmoid(logits) - (1-p) * jax.nn.log_sigmoid(-logits)
            return (mus[:, 0, 0:1] * (-kappa * p + s * entropy)).sum(axis=1)

        return population, payoff, lambda logits, gain: kappa + s * logits - gain, fit
    return build
