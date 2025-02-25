import sys
import numpy as np
import matplotlib.pyplot as plt

def empty(x, y):
    return x + y

def sphere(x, y):
    return x**2 + y**2

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def matyas(x, y):
    return 0.26*(x**2 + y**2) - 0.48*(x*y)

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def rastrigin(x, y):
    return 20 + x**2 + y**2 - 10*(np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

def easom(*args):
    result = -np.prod([np.cos(x) for x in args]) * np.exp(-np.sum([(x - np.pi) ** 2 for x in args]))
    return result


def plot_function(func, xlim=(-5, 5), ylim=(-5, 5)):
    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def plot_kindeas(func, n, lim=(-5, 5)):
    x = np.random.randint(lim[0], lim[1], size=(n))
    result = func(*x)
    print(f"Wartość dla {n}-wymiaru podanych punktow: {x} wynosi:\n {result}")

if __name__ == "__main__":
    function_name = sys.argv[1]
    if function_name == "empty":
        plot_function(empty)
    elif function_name == "sphere":
        plot_function(sphere)
    elif function_name == "rosenbrock":
        plot_function(rosenbrock)
    elif function_name == "beale":
        plot_function(beale)
    elif function_name == "matyas":
        plot_function(matyas)
    elif function_name == "rastrigin":
        plot_function(rastrigin)
    elif function_name == "easom":
        n = int(sys.argv[2])
        plot_kindeas(easom, n)
    else:
        print("\nPodaj poprawną nazwę funkcji.")
