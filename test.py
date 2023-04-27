import random

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt



def data():
    cnt = 0
    x = []
    y = []
        
    for i in  range(1,10):
        # x = []
        # y = []
        x.append(cnt*i)
        y.append(random.randint(0, 10))
        
        cnt += 1  
        yield  x, y, cnt
    
    input('any key to exit !!!')
    quit()

def init_animate():
    pass

def animate( data, *fargs) :
    print('data : ', data, '\n data type : ', type(data), ' cnt : ', data[2])
    plt.cla()
    x = [i*k for i in data[0]]
    y = [i*p for i in data[1]]
    
    plt.plot(x,y)


if __name__ == "__main__":
    fig = plt.figure()
    k = 3
    p = 5
    ani = FuncAnimation(fig, animate, init_func=init_animate, frames=data,  interval=700, fargs = [k,p])
    plt.show()