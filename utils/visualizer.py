import visdom
import numpy as np

class Visualizer:
    def __init__(self, env_name='main', port=8097):
        self.vis = visdom.Visdom(env=env_name, port=port)

    def plot_line(self, X, Y, win, opts):
        if not self.vis.line_exists(win=win):
            self.vis.line(X=X, Y=Y, win=win, opts=opts)
        else:
            self.vis.line(X=X, Y=Y, win=win, opts=opts, update='append')

    def plot_image(self, image, opts):
        self.vis.image(image, opts=opts)

if __name__=='__main__':
    def main():
        vis=Visualizer()
        X=np.array([0,1,2,3])
        Y=np.array([0,1,4,9]) 
        vis.plot_line(X=X, Y=Y, win='example',opts=dict(title='Line Plot'))  
    main()    
