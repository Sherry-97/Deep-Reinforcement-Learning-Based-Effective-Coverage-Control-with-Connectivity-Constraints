import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario




class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
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
            agent.state.p_pos = np.random.uniform(-10, +10, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.connect = 0
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-10, +10, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.energy = 0
            landmark.m = 0
            landmark.step = 120

    def benchmark_data(self, agent, world):
        
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
           
                
        if coverage:
            coverage_number += 1
        coverage_rate = coverage_number/20
        ce = self.connection(world.agents[0], world.agents[1], world.agents[2])
        if ce < 0.00001:
             rew -= 1
      
        return (rew, coverage_rate)
   


    def connection(self, agent0, agent1, agent2): 
        n = 4       
        ce = eig(laplacian_matrix[1])
        return ce
       
       

    
    def reward(self, agent, world):
        

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
           
                
        if coverage:
            coverage_number += 1
        coverage_rate = coverage_number/20
        ce = self.connection(world.agents[0], world.agents[1], world.agents[2])
        if ce < 0.00001:
             rew -= 1
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


