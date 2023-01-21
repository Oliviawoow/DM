import numpy as np 
import csv

# G = link structure of a network of web pages
# d = default damping factor 
# epsilon = default threshold
def page_rank(G, d=0.85, epsilon=10e-8):
    # gets the number of rows in the input matrix G
    n = G.shape[0]
    # creates an nxn matrix "M" filled with zeros
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # creates probability matrix M if there is a link between (i,j)
            if G[j, i] != 0:
                M[i, j] = 1/G[j, i]
    # creates a vector v with dimensions nx1, filled with 1/n
    v = np.ones((n, 1))/n
    last_v = np.ones((n, 1)) * 10e6
    # matrix used in the PageRank calculation
    M_hat = (d * M) + (((1-d)/n) * np.ones((n, n)))
    # will run until the difference between the current score vector v 
    # and the previous score vector last_v is less than the threshold epsilon
    while np.linalg.norm(v - last_v) > epsilon:
        last_v = v
        v = np.matmul(M_hat, v)
        if np.linalg.norm(v - last_v) == np.Inf:
            return v
    # returns the final score vector v which represents the PageRank scores for all pages in the network
    return v

matrix = []
with open("/home/ollie/Downloads/PageRank/dataset3.csv", 'r') as f:
    reader = csv.reader(f, delimiter=',')             
    for row in reader:
        matrix_row = [int(x) for x in row]
        matrix.append(matrix_row)

G = np.array(matrix)
rez = page_rank(G)
print(rez)