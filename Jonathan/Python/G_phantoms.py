import numpy as np

d = 91
w = 20

def remove_outside(G, outside=0.5, d=91):
    out = G.copy()
    r = d//2
    for i in range(d):
        for j in range(d):
            if (i-r)**2 + (j-r)**2 > r**2:
                out[i+j*d] = outside
    return out

def gen_G_borders_rl(d=91, w=20):
    G_borders_rl = np.zeros(d**2)
    for j in range(0,d):
        for i in range(0,w):
            G_borders_rl[i+d*j] = 1
            G_borders_rl[d-i-1+d*j] = 1
    return G_borders_rl
G_borders_rl = gen_G_borders_rl(d, w)
G_borders_rl = remove_outside(G_borders_rl)

def gen_G_square(d=91, w=20):
    G_square = np.zeros(d**2)
    for i in range(d//2-w,d//2+w):
        for j in range(d//2-w,d//2+w):
            G_square[i+d*j] = 1
    return G_square
G_square = gen_G_square(d, w)
G_square = remove_outside(G_square)

def gen_G_circle(d=91, w=20):
    G_circle = np.zeros(d**2)
    for i in range(d):
        for j in range(d):
            if (i-d//5)**2 + (j-d//5)**2 < (d//5)**2:
                G_circle[i+j*d] = 1
    return G_circle
G_circle = gen_G_circle(d, w)
G_circle = remove_outside(G_circle)

def gen_G_heart(d=91, w=20):
    G_heart = np.zeros(d**2)
    for i in range(d):
        for j in range(d):
            x = (i-d//2)/w*0.75
            y = (d//2-j)/w+0.4
            b = 1-x**2
            if b >= 0:
                a = (x**2)**(1/3)
                b = np.sqrt(b)
                if y >= a-b and y <= a+b:
                    G_heart[i+j*d] = 1
    return G_heart
G_heart = gen_G_heart(d, w)
G_heart = remove_outside(G_heart)

def gen_G_hidden(d=91, w=20):
    G_hidden = np.zeros(d**2)
    r_outer = ((d-w//2)/2)**2
    r_inner = (w)**2
    for i in range(d):
        for j in range(d):
            r = (i - d//2)**2 + (d//2 - j)**2
            if r > r_outer:
                G_hidden[i+j*d] = 1
            elif r < r_inner:
                G_hidden[i+j*d] = 1
    return G_hidden
G_hidden = gen_G_hidden(d, w)
G_hidden = remove_outside(G_hidden)
