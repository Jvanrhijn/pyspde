from src.spde import DerivativeOperator
import numpy as np


def linear(points, k, sigma):
    return lambda a, t, w: -k**2 * a + sigma*w


def geometric_brownian(points, k, sigma, left, right):
    d1 = DerivativeOperator(1, points, left, right)
    return lambda a, t, w: - d1(a)**2/a + a*sigma*w


def linmult_arnold(points, k, sigma, left, right):
    d1 = DerivativeOperator(1, points, left, right)
    return lambda a, t, w: -d1(a)**2 * a/(1 + a**2) - (k - sigma**2)**2 * a / (1 + a**2) + sigma * np.sqrt(1 + a**2)*w


def linmult_graham(points, k, sigma, left, right):
    d1 = DerivativeOperator(1, points, left, right)
    return lambda a, t, w: -d1(a)**2 * a/(1 + a**2) - ((k - sigma**2)**2 - sigma**4/4) * a / (1 + a**2) + sigma * np.sqrt(1 + a**2)*w


def gaussian_arnold(points, k, sigma, left, right):
    d1 = DerivativeOperator(1, points, left, right)
    return lambda a, t, w: a*d1(a)**2 + sigma**4 / 4 * np.exp(-2*a**2) * (a**3 - 1.5*a) + sigma * np.exp(-a**2/2)*w
