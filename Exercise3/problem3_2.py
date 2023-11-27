def diffMatrix(u, v):
    M = []
    for u_i in u:
        row = []
        for v_j in v:
            row.append(u_i - v_j)
        M.append(row)
    return M

def prodMatrix(u, v):
    M = []
    for u_i in u:
        row = []
        for v_j in v:
            row.append(u_i * v_j)
        M.append(row)
    return M

if __name__ == '__main__':
    u = [1, 2, 3]
    v = [4, 5, 6, 7]
    print(diffMatrix(u, v))
    print(prodMatrix(u, v))
