# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np


cdef extern from "Vector2.h" namespace "RVO":
    cdef cppclass Vector2:
        Vector2() except +
        Vector2(float x, float y) except +
        float x() const
        float y() const

cdef extern from "Agent.h" namespace "RVO":
    cdef cppclass Line:
        Vector2 point
        Vector2 direction

cdef extern from "Agent.h" namespace "RVO":
    cdef cppclass Agent:
        Agent(const Vector2 &position, float neighborDist, size_t maxNeighbors, float timeHorizon,float timeHorizonObs, float radius, float maxSpeed, const Vector2 &velocity) except +
        void computeNeighbors()
        void setAgentPrefVelocity(const Vector2 &prefVelocity)
        const Vector2 & getAgentVelocity() const
        const Vector2 & getAgentPosition() const
        const vector[Line] &getAgentORCALines() const
        void computeNewVelocity()
        void computeNewVelocity1()
        void insertAgentNeighbor(const Agent *agent, float rangeSq)
        void loadAgentNeighbors(const vector[Vector2] &positions, const vector[Vector2] &velocities)
        void self_update(const Vector2 &position,const Vector2 &velocity)


cdef class ORCA_agent:
    cdef Agent *agent
    def __cinit__(self, tuple pos, float neighborDist,
                 size_t maxNeighbors, float timeHorizon,
                 float radius, float maxSpeed,
                 tuple velocity=(0,0)):
        cdef Vector2 c_pos = Vector2(pos[0], pos[1])
        cdef Vector2 c_velocity

        c_velocity = Vector2(velocity[0], velocity[1])
        self.agent = new Agent(c_pos, neighborDist,
                                             maxNeighbors, timeHorizon, 0,
                                             radius, maxSpeed,
                                             c_velocity)

    def setAgentPrefVelocity(self, tuple velocity):
        cdef Vector2 c_velocity = Vector2(velocity[0], velocity[1])
        self.agent.setAgentPrefVelocity(c_velocity)

    # def get_vel_pos(self):
    #     #return current velocity and position
    #     cdef Vector2 vel = self.agent.getAgentVelocity()
    #     cdef Vector2 pos = self.agent.getAgentPosition()
    #     return np.array([vel.x(),vel.y()]), np.array([pos.x(),pos.y()])

    def get_orca_lines(self):
        cdef vector[Line] lines = self.agent.getAgentORCALines()
        collection_lines = []
        for i in range(lines.size()): 
          line = np.array([lines[i].point.x(),lines[i].point.y(),lines[i].direction.x(),lines[i].direction.y()])
          collection_lines.append(line)
        return np.array(collection_lines)          
	

    def update_neighbour_states(self, np.ndarray positions_n, np.ndarray velocities_n):
        cdef vector[Vector2] positions
        cdef vector[Vector2] velocities

        cdef int neighbor_num = positions_n.shape[0]

        for i in range(neighbor_num):
            positions.push_back(Vector2(positions_n[i,0],positions_n[i,1]))
            velocities.push_back(Vector2(velocities_n[i,0],velocities_n[i,1]))

        self.agent.loadAgentNeighbors(positions, velocities)
        self.agent.computeNeighbors()

    def self_update(self, tuple pos, tuple vel):
        cdef Vector2 position = Vector2(pos[0], pos[1]) 
        cdef Vector2 velocity = Vector2(vel[0], vel[1]) 

        self.agent.self_update(position,velocity)

    def computeNewVelocity(self):
    
        self.agent.computeNewVelocity()
        # cdef Vector2 velocity1 = self.agent.getAgentVelocity()

        # return (velocity1.x(), velocity1.y())
    
    def computeNewVelocity1(self):
    
        self.agent.computeNewVelocity1()
        cdef Vector2 velocity1 = self.agent.getAgentVelocity()

        return (velocity1.x(), velocity1.y())
        



