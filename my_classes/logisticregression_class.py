import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import seaborn as sns
import matplotlib.gridspec as gridspec

class Logisticregression():

    def __init__(self, learning_rate=0.01, epochs=100, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.hist = []
        self.costs = []
        
    def X_B(self, X):
        return self.beta[0] + np.dot(X, self.beta[1:])
    
    def P(self, z):
        return 1./(1.+np.exp(-z))
    
    def predict(self, X):
        proba = self.P(self.X_B(X))
        return np.where(proba >= self.threshold, 1, 0)
    
    def cost(self, y, proba):
        return np.dot(y,np.log(proba)) + np.dot((1-y),np.log(1-proba))
        #return np.dot(y,self.X_B(X)) - np.log(1 + np.exp(self.X_B(X))) 
    
    def gradient_decent(self, X, y):
        for i in range(self.epochs):
            proba = self.P(self.X_B(X))
            proba = np.where(proba < 1e-6, 1e-6, proba) #avoid problem with log giving -infinity
            proba = np.where(proba > 1 - 1e-6, 1 - 1e-6, proba) #avoid problem with log giving -infinity
            errors = y - proba
            self.beta[1:] += self.learning_rate * X.T.dot(errors)
            self.beta[0] += self.learning_rate * errors.sum()
            yield self.cost(y, proba)

    def fit(self, X, y):
        self.beta = np.random.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])
        self.hist.append(self.beta.copy())

        for cost in self.gradient_decent(X, y):
            self.hist.append(self.beta.copy())
            self.costs.append(cost)
        return self
    
    def Logistic_graph(self,X,xylim=10,name=None,path=".\my_visualization"):
        interval = np.linspace(-abs(xylim),abs(xylim),10000)
        logistic = self.P(interval)

        fig = plt.figure() 
        axis = plt.axes(xlim =(-abs(xylim),abs(xylim)), 
                        ylim =(-0.5, 1.5)) 
        axis.plot(interval, logistic, 'k--', label="Sigmoid base")
        annot = axis.annotate('Epoch: 0', (0.09, 0.92), xycoords='figure fraction')
        line, = axis.plot([], [], 'ro') 
        

        def init(): 
            line.set_data([], []) 
            return line,

        def animate(frame):
            Beta = self.hist[frame] 
            abcissa = Beta[0] + np.dot(X, Beta[1:])
            ordinate = self.P(abcissa)
            line.set_data(abcissa, ordinate) 
            annot.set_text(f'Epoch: {frame}')
            
            return line, annot
        
        ani = FuncAnimation(fig, animate, frames=len(self.hist), init_func=init,
                            blit=False, interval=200, repeat=False)
        if isinstance(name,str):
            ani.save(f"{path}\{name}.gif")
            print(f"image saved as {name} at the path : {path}")
        plt.legend()
        plt.show()
        
        return
    
    def Logistic_heatmap(self,name=None,path=".\my_visualization"):
        # Reshape betas (skip bias term) into 8x8 grids
        list_mat = [mat[1:].reshape(8, 8) for mat in self.hist]

        grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
        fig, (ax,cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (8, 8))

        def init_heatmap():
            fig, (ax,cbar_ax) = plt.subplots(1, 2, figsize = (8, 8))
            sns.heatmap(list_mat[0], ax=ax, cbar=False, annot=True)
            return fig, ax, cbar_ax
        
        def animate(j):
            ax.clear()  # clear old heatmap
            cbar_ax.clear()
            sns.heatmap(list_mat[j], cbar=True, cbar_ax=cbar_ax, square=True, ax=ax, annot=True)
            title = ax.set_title(f"Epoch: {j}",loc='left')
            return ax, cbar_ax

        anim = FuncAnimation(fig, animate, frames=len(list_mat),
                            interval=300, blit=False, repeat=False)
        if isinstance(name,str):
            anim.save(f"{path}\{name}.gif")
            print(f"image saved as {name} at the path : {path}")
        plt.show()
        return
    
    def Logistic_heatmap_with_costs(self, name=None, path="./my_visualization"):
        # Reshape betas (skip bias term) into 8x8 grids
        list_mat = [mat[1:].reshape(8, 8) for mat in self.hist]

        # Fix color scale across all epochs
        vmin = min(mat.min() for mat in list_mat)
        vmax = max(mat.max() for mat in list_mat)

        # Create figure and GridSpec layout
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[0.90, 0.05], height_ratios=[0.8, 0.2], wspace=0.1, hspace=0.1)

        ax_1 = fig.add_subplot(gs[0, 0])    # Heatmap
        cbar_ax = fig.add_subplot(gs[0, 1]) # Colorbar
        ax_2 = fig.add_subplot(gs[1, :])    # Cost plot spanning bottom row

        def init_plot():
            # Initial heatmap
            sns.heatmap(list_mat[0], ax=ax_1, cbar=False, annot=True)
            ax_1.set_title("Epoch: 0", loc="left")

            # Initial cost plot
            ax_2.plot([], [])
            ax_2.set_xlim(0, len(self.costs))
            ax_2.set_ylim(min(self.costs), max(self.costs)+(1/2)*abs(max(self.costs)-min(self.costs)))
            ax_2.set_xlabel("Epoch")
            ax_2.set_ylabel("Cost")

        def animate(j):
            ax_1.clear()  # clear old heatmap
            cbar_ax.clear()
            ax_2.clear()
            # Update heatmap data
            sns.heatmap(list_mat[j], cbar=True, cbar_ax=cbar_ax, square=True, ax=ax_1, annot=True)
            ax_1.set_title(f"Epoch: {j}", loc="left")

            # Update cost plot up to current epoch
            y = self.costs
            x = [i+1 for i in range(len(y))]

            ax_2.plot(x[:j+1], y[:j+1], color="black")
            ax_2.set_xlim(0, len(self.costs))
            ax_2.set_ylim(min(self.costs), max(self.costs)+(1/2)*abs(max(self.costs)-min(self.costs)))
            ax_2.set_xlabel("Epoch")
            ax_2.set_ylabel("Cost")
            return ax_1, cbar_ax, ax_2

        anim = FuncAnimation(fig, animate, frames=len(list_mat),
                            interval=300, blit=False, repeat=False)

        # Save animation if requested
        if isinstance(name, str):
            anim.save(f"{path}\{name}.gif")
            print(f"Image saved as {name}.gif at path: {path}")

        plt.show()
        return anim