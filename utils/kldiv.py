
P = open('../testDump1.txt', 'rt').read().split()
P = [float(p) for p in P]

Q = open('../testDump2.txt', 'rt').read().split()
Q = [float(q) for q in Q]

print(min(P), max(P))
print(min(Q), max(Q))
exit(1)

for p, q in zip(P, Q):
    print(p, q, p/q)
    if p/q < 0:
        break