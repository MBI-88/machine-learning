#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gym

env=gym.make('CartPole-v1')
env.observation_space


# In[2]:


env.action_space


# In[3]:


env.reset()


# In[4]:


env.step(action=0)


# In[5]:


env.render()


# In[6]:


env.close()


# In[7]:


env.step(action=1)


# In[8]:


env.render()


# In[9]:


env.close()


# # Grid World example

# In[10]:


import numpy as np 
from gym.envs.toy_text import discrete
from collections import defaultdict
import time 
import pickle 
import os
from gym.envs.classic_control import rendering 


# In[11]:


Cell_size=100
Margin=10

def get_coords(row,col,loc='center'):
    xc=(col+1.5)*Cell_size 
    yc=(row+1.5)*Cell_size

    if loc == 'center':
        return xc,yc 
    
    elif loc=='interior_corners':
        half_size=Cell_size//2-Margin
        xl,xr=xc-half_size,xc+half_size
        yt,yb=xc-half_size,xc+half_size
        return [(xl,yt),(xr,yt),(xr,yb),(xl,yb)]
    
    elif loc == 'interior_triangle':
        x1,y1=xc,yc+Cell_size//3
        x2,y2=xc+Cell_size//3,yc-Cell_size//3
        x3,y3=xc-Cell_size//3,yc-Cell_size//3
        return [(x1,y1),(x2,y2),(x3,y3)]
    
def draw_objetc(coords_list):
    if len(coords_list)==1:
        obj=rendering.make_circle(int(0.45*Cell_size))
        obj_transform=rendering.Transform()
        obj.add_attr(obj_transform)
        obj_transform.set_translation(*coords_list[0])
        obj.set_color(0.2,0.2,0.2) # negro
    elif len(coords_list)==3:
        obj=rendering.FilledPolygon(coords_list)
        obj.set_color(0.9,0.6,0.2) # amarillo
    elif len(coords_list) > 3:
        obj=rendering.FilledPolygon(coords_list)
        obj.set_color(0.4,0.4,0.8) # azul
    return obj


# In[12]:


# Script GridWorldEnv

class GridWorldEnv(discrete.DiscreteEnv):
    def __init__(self,num_rows=4,num_cols=6,delay=0.05):
        self.num_rows=num_rows
        self.num_cols=num_cols
        self.delay=delay

        move_up=lambda row,col: (max(row-1,0),col)
        move_down=lambda row,col: (min(row+1,num_rows-1),col)
        move_left=lambda row,col: (row,max(col-1,0))
        move_right=lambda row,col:(row,min(col+1,num_cols-1))
        self.action_defs={0:move_up,1:move_right,2:move_down,3:move_left}

        # Numeros de estados / accion
        nS=num_cols*num_rows
        nA=len(self.action_defs)
        self.grid2state_dict={(s//num_cols,s%num_cols):s for s in range(nS)}
        self.state2grid_dict={s:(s//num_cols,s%num_cols)  for s in range(nS)}

        # Estado objetivo
        gold_cell=(num_rows//2,num_cols-2)

        # Estado  trampas
        trap_cells=[((gold_cell[0]+1),gold_cell[1]),
                    (gold_cell[0],gold_cell[1]-1),
                    ((gold_cell[0]-1),gold_cell[1])]
        
        gold_state=self.grid2state_dict[gold_cell]
        trap_states=[self.grid2state_dict[(r,c)] for (r,c) in trap_cells]
        self.terminal_states=[gold_state]+trap_states
        print(self.terminal_states)

        # Crando las probabilidades de transicion
        P=defaultdict(dict)
        for s in range(nS):
            row,col=self.state2grid_dict[s]
            P[s]=defaultdict(list)
            for a in range(nA):
                action=self.action_defs[a]
                next_s=self.grid2state_dict[action(row,col)]

                # Estado terminal
                if self.is_terminal(next_s):
                    r=(1.0 if next_s==self.terminal_states[0] else -1.0)
                else:
                    r=0.0
                if self.is_terminal(s):
                    done=True
                    next_s=s
                else:
                    done=False
                P[s][a]=[(1.0,next_s,r,done)]

        # Distribucion de estados iniciales
        isd=np.zeros(nS)
        isd[0]=1.0

        super(GridWorldEnv,self).__init__(nS,nA,P,isd)
        self.viewer=None
        self._build_display(gold_cell,trap_cells)
    
    def is_terminal(self,state):
        return state in self.terminal_states
    
    def _build_display(self,gold_cell,trap_cells):

        screen_width=(self.num_cols+2)*Cell_size
        screen_height=(self.num_rows+2)*Cell_size
        self.viewer=rendering.Viewer(screen_width,screen_height)

        all_objects=[]

        # Lista de puntos de bordes cordenados
        bp_list=[
            (Cell_size-Margin,Cell_size-Margin),
            (screen_width-Cell_size+Margin,Cell_size-Margin),
            (screen_width-Cell_size+Margin,screen_height-Cell_size+Margin),
            (Cell_size-Margin,screen_height-Cell_size+Margin)
        ]

        border=rendering.PolyLine(bp_list,True)
        border.set_linewidth(5)
        all_objects.append(border)

        # Lineas Verticales

        for col in range(self.num_cols+1):
            x1,y1=(col+1)*Cell_size,Cell_size
            x2,y2=(col+1)*Cell_size,(self.num_rows+1)*Cell_size
            line=rendering.PolyLine([(x1,y1),(x2,y2)],False)
            all_objects.append(line)
        
        # Linea horizontales

        for row in range(self.num_rows+1):
            x1,y1=Cell_size,(row+1)*Cell_size
            x2,y2=(self.num_cols+1)*Cell_size,(row+1)*Cell_size
            line=rendering.PolyLine([(x1,y1),(x2,y2)],False)
            all_objects.append(line)
        
        # Traps: circles
        for cell in trap_cells:
            trap_coords=get_coords(*cell,loc='center')
            all_objects.append(draw_objetc([trap_coords]))
        
        # Gold: triangle
        gold_coords=get_coords(*gold_cell,loc='interior_triangle')
        all_objects.append(draw_objetc(gold_coords))

        # Agent square or robot
        if (os.path.exists('robot-coordinates.pkl')and Cell_size==100):
            agent_coords=pickle.load(open('robot-coordinates.pkl','rb'))
            starting_coords=get_coords(0,0,loc='center')
            agent_coords += np.array(starting_coords)
        else:
            agent_coords=get_coords(0,0,loc='interior_corners')
        agent=draw_objetc(agent_coords)
        self.agent_trans=rendering.Transform()
        agent.add_attr(self.agent_trans)
        all_objects.append(agent)

        for obj in all_objects:
            self.viewer.add_geom(obj)
        
    def render(self,mode='human',done=False):
        if done:
            sleep_time=1
        else:
            sleep_time=self.delay
        x_coord=self.s % self.num_cols
        y_coord=self.s // self.num_cols
        x_coord=(x_coord+0)*Cell_size
        y_coord=(y_coord+0)*Cell_size
        self.agent_trans.set_translation(x_coord,y_coord)
        rend=self.viewer.render(
            return_rgb_array=(mode=='rgb_array'))
        
        time.sleep(sleep_time)
        return rend
    

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer=None


# In[13]:


if __name__=='__main__':
    env=GridWorldEnv(5,6)
    for i in range(1):
        s=env.reset()
        env.render(mode='human',done=False)

        while True:
            action=np.random.choice(env.nA)
            res=env.step(action)
            print('Action ',env.s,action,' ->  ',res)
            env.render(mode='human',done=res[2])
            if res[2]:
                break
    env.close()


# # Grid World example con Q-Learning

# In[14]:


# Script agente

class Agent(object):
    def __init__(self,env,learning_rate=0.01,discount_factor=0.9,epsilon_greedy=0.9,epsilon_min=0.1,epsilon_decay=0.95):
        self.env=env
        self.lr=learning_rate
        self.gamma=discount_factor
        self.epsilon=epsilon_greedy
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay

        # Definicion de la tabla q
        self.q_table=defaultdict(lambda:np.zeros(self.env.nA))  
    
    def choose_action(self,state):
        if np.random.uniform() < self.epsilon:
            action=np.random.choice(self.env.nA)
        else:
            q_vals=self.q_table[state]
            perm_actions=np.random.permutation(self.env.nA)
            q_vals=[q_vals[a] for a in perm_actions]
            perm_q_argmax=np.argmax(q_vals)
            action=perm_actions[perm_q_argmax]
        return action
    
    def _learn(self,transition):
        s,a,r,next_s,done=transition
        q_vals=self.q_table[s][a]
        if done:
            q_target=r
        else:
            q_target=r+self.gamma*np.max(self.q_table[next_s])
        
        # Actualizando la tabla
        self.q_table[s][a]+=self.lr*(q_target-q_vals)

        # Ajustando epsilon
        self._adjust_epsilon()
    
    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        


# In[15]:


# Script qlearning
from collections import namedtuple
import matplotlib.pyplot as plt 
np.random.seed(1)

Transition=namedtuple(
    'Transition',('state','action','reward','next_state','done'))

def run_qlearning(agent,env,num_episodes=50):
    history=[]
    for episodes in range(num_episodes):
        state=env.reset()
        env.render(mode='human')
        final_reward,n_moves=0.0,0
        while True:
            action=agent.choose_action(state)
            next_s,reward,done,_=env.step(action)
            agent._learn(Transition(state,action,reward,next_s,done))
            env.render(mode='human',done=done)
            state=next_s
            n_moves += 1
            if done:
                break
            final_reward=reward
        history.append((n_moves,final_reward))
        print('Episode %d: Rward %.1f #Moves %d'% (episodes,final_reward,n_moves))
    return history

def plot_learning_history(history):
    fig=plt.figure(1,figsize=(14,10))
    ax=fig.add_subplot(2,1,1)
    episodes=np.arange(len(history))
    moves=np.array([h[0] for h in history])
    plt.plot(episodes,moves,lw=4,marker='o',markersize=10)

    ax.tick_params(axis='both',which='major',labelsize=15)
    plt.xlabel('Episodes',size=20)
    plt.ylabel('# moves',size=20)

    ax=fig.add_subplot(2,1,2)
    rewards=np.array([h[1]for h in history])
    plt.step(episodes,rewards,lw=4)
    ax.tick_params(axis='both',which='major',labelsize=15)
    plt.ylabel('Final rewards',size=20)
    plt.xlabel('Episodes',size=20)
    plt.savefig('q-learning-history.png',dpi=300)
    plt.show()


# In[16]:


if __name__=='__main__':
    env=GridWorldEnv(num_rows=5,num_cols=6)
    agent=Agent(env)
    history=run_qlearning(agent,env)
    env.close()

    plot_learning_history(history)


# # Deep Q-Learning

# In[17]:


import gym 
import numpy as np 
import tensorflow as tf 
import random 
import matplotlib.pyplot as plt 
from collections import namedtuple,deque
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(1)
tf.random.set_seed(1)


# In[18]:


Transition=namedtuple('Transition',('state','action','reward','next_state','done'))

class DQNAgent(object):
    def __init__(self,env,discount_factor=0.95,epsilon_greedy=1.0,epsilon_min=0.01,epsilon_decay=0.995,learning_rate=1e-3,max_memory_size=2000):
        self.enf=env
        self.state_size=env.observation_space.shape[0]
        self.action_size=env.action_space.n

        self.memory=deque(maxlen=max_memory_size)

        self.gamma=discount_factor
        self.epsilon=epsilon_greedy
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay
        self.lr=learning_rate
        self._build_nn_model()

    def _build_nn_model(self,n_layers=3):
        # Capas ocultas
        self.model=tf.keras.Sequential()

        for n in range(n_layers-1):
            self.model.add(tf.keras.layers.Dense(units=32,activation='relu'))
            self.model.add(tf.keras.layers.Dense(units=32,activation='relu'))
        
        # Last layer
        self.model.add(tf.keras.layers.Dense(units=self.action_size))

        # Creacion y compilacion
        self.model.build(input_shape=(None,self.state_size))
        self.model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

    def remember(self,transition):
        self.memory.append(transition)
        
    def choose_action(self,state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        q_values=self.model.predict(state)[0]
        return np.argmax(q_values)
    
    def _learn(self,batch_sample):
        batch_state,batch_target=[],[]
        for transition in batch_sample:
            s,a,r,next_s,done=transition
            if done:
                target=r
            else:
                target=(r+self.gamma*np.amax(self.model.predict(next_s)[0]))
            target_all=self.model.predict(s)[0]
            target_all[a]=target
            batch_state.append(s.flatten())
            batch_target.append(target_all)
            self._adjust_epsilon()
        return self.model.fit(x=np.array(batch_state),y=np.array(batch_target))
    
    def replay(self,batch_size):
        samples=random.sample(self.memory,batch_size)
        history=self._learn(samples)
        return history.history['loss'][0]

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *=  self.epsilon_decay
        
    

def plot_learning_history(history):
    fig=plt.figure(1,figsize=(14,5))
    ax=fig.add_subplot(1,1,1)
    episodes=np.arange(len(history))+1
    plt.plot(episodes,history,lw=4,marker='o',markersize=10)
    ax.tick_params(axis='both',which='major',labelsize=15)
    plt.xlabel('Episodes',size=20)
    plt.ylabel('# Total Rewards',size=20)
    plt.show()


# In[19]:


Episodes=10 # Por cuestion de tiempo
batch_size=32
init_replay_memory_size=500

if __name__=='__main__':
    env=gym.make('CartPole-v1')
    agent=DQNAgent(env)
    state=env.reset()
    state=np.reshape(state,[1,agent.state_size])

    # Rellenando la memoria
    for i in range(init_replay_memory_size):
        action=agent.choose_action(state)
        next_state,reward,done,_=env.step(action)
        next_state=np.reshape(next_state,[1,agent.state_size])
        agent.remember(Transition(state,action,reward,next_state,done))

        if done:
            state=env.reset()
            state=np.reshape(state,[1,agent.state_size])
        else:
            state=next_state

    total_rewards,losses=[],[]
    for e in range(Episodes):
        state=env.reset()
        if  e % 10==0:
            env.reset()
        state=np.reshape(state,[1,agent.state_size])
        for i in range(500):
            action=agent.choose_action(state)
            next_state,reward,done,_=env.step(action)
            next_state=np.reshape(next_state,[1,agent.state_size])

            agent.remember(Transition(state,action,reward,next_state,done))
            state=next_state
            if e % 10==0:
                env.render()
            if done:
                total_rewards.append(i)
                print('Episode:  %d/%d, Total reward: %d'% (e,Episodes,i))
                break
            loss=agent.replay(batch_size)
            losses.append(loss)
    env.close()
    plot_learning_history(total_rewards)


# In[ ]:




