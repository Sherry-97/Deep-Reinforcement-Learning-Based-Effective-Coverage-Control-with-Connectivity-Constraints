import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario




class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 20
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.8
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.connect = 0
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.energy = 0
            landmark.m = 0
            landmark.step = 120

    def benchmark_data(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions

        cover_number = 0
        rew = 0.0

        for l in world.landmarks:

            for k in world.agents:
                dist = np.sqrt(np.sum(np.square(k.state.p_pos - l.state.p_pos)))
                if dist > agent.size:
                    power = 0.0
                else:
                    power = 1.0 / np.power(agent.size, 4) * np.power((np.power(dist, 2) - np.power(agent.size, 2)), 2)
                l.energy += power
            if l.energy > 3.2:
                rew += 0.1
                cover_number += 1

        l.m += 1
        coverage_rate = cover_number * 5

        ce = self.connection(world.agents[0], world.agents[1], world.agents[2])
        if ce < 0.00001:
            # print("True")
            # rew -= 0.05
            k.connect += 1

            # print("False")

        if coverage_rate == 100:
            l.step = min(l.step, l.m / 3)

        if l.m == 360:
            print(
                'coverage_rate: %lf connectivity: %lf step: %d' % (coverage_rate / 100, 1 - (k.connect / 360), l.step),
                end=' ')
        return (rew, coverage_rate, k.connect)
   


    def connection(self, agent0, agent1, agent2): 
        n = 3       
        def bound(x):
            if x > 2:
                return 0
            else:
                return 1
        delta_pos_1 = agent0.state.p_pos - agent1.state.p_pos
        dist1 = np.sqrt(np.sum(np.square(delta_pos_1)))
        delta_pos_2 = agent0.state.p_pos - agent2.state.p_pos
        dist2 = np.sqrt(np.sum(np.square(delta_pos_2)))
        delta_pos_3 = agent1.state.p_pos - agent2.state.p_pos
        dist3 = np.sqrt(np.sum(np.square(delta_pos_3)))


       
        G = np.array([None] * n)
        D = np.array([None] * n)
        for i in range(len(G)):
            G = np.array([[0] * n] * n)
            D = np.array([[0] * n] * n)
        G[0][1] = bound(dist1)
        G[1][0] = bound(dist1)
        G[0][2] = bound(dist2)
        G[2][0] = bound(dist2)
        G[2][1] = bound(dist3)
        G[1][2] = bound(dist3)
        D[0][0] = G[0][1] + G[0][2]
        D[1][1] = G[1][0] + G[1][2]
        D[2][2] = G[2][0] + G[2][1]
        L = D - G
        eigenvalue, eigenvector = np.linalg.eig(L)
        eig = np.sort(eigenvalue)
        ce = eig[1]
        return ce
       
       

    
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions

        cover_number = 0
        rew = 0.0
        
        
        for l in world.landmarks:
            
            for k in world.agents:
                dist = np.sqrt(np.sum(np.square(k.state.p_pos - l.state.p_pos))) 
                if dist > agent.size:
                    power = 0.0
                else:
                    power = 1.0 / np.power(agent.size,4) * np.power((np.power(dist,2) - np.power(agent.size,2)),2)
                l.energy += power  
            if l.energy > 3.2:
                rew += 0.1
                cover_number += 1
            
        l.m += 1      
        coverage_rate = cover_number * 5
        
        
            
    
        
        ce = self.connection(world.agents[0], world.agents[1], world.agents[2])
        if ce < 0.00001:
            #print("True")
            #rew -= 0.05
            k.connect += 1


        
            #print("False")

        if coverage_rate == 100:
            l.step = min(l.step, l.m / 3)

        if l.m == 360:
            print('coverage_rate: %lf connectivity: %lf step: %d' % (coverage_rate / 100, 1 - (k.connect / 360), l.step), end = ' ')
        return rew
   



    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)


