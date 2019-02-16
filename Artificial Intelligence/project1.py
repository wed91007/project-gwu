# -*- coding: utf-8 -*-
import numpy as np
import heapq
from collections import defaultdict
import common as conf
edges=list()
vertices=list()
count=0
countE=0
#read data
#The data must contain 'vertices' and 'edges' to make function understand
def readfile(filepath):
    global vertices
    global edges
    f = open(filepath,'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        if line == 'Vertices':
            continue    
        if line == 'Edges':
           break
        vertices.append(line)
    f.close()
    f = open(filepath,'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        
        if line == 'Vertices':
            continue
        
        if line == 'Edges':
            continue
        edges.append(line)
    f.close()        
    edges =list(set(edges).difference(set(vertices)))
    edges.sort()         


#define graph
#class graph:
#    def __init__(self,vertices,edges):
##
def adjacencyMat(vertices,edges):
    global count
    global countE
    admat = np.full((count,count),np.inf)
    for i in range(0,countE):
        x=int(edges[i][0])
        y=int(edges[i][1])
        cost=int(edges[i][2])
        admat[y][x]=admat[x][y] = cost
    return admat
        
#convert str list into numpy array in int        
def regular(vertices,edges):    
    global count
    global countE
    v = np.zeros((count,3),dtype=np.int)
    e = np.zeros((countE,3),dtype=np.int)
    
    for i in range(0,count):
        a = vertices[i].split(',')
        for j in (0,1,2):
            v[i][j]=a[j]
            
    for i in range(0,countE):
        b = edges[i].split(',')
        for j in (0,1,2):
            e[i][j]=b[j]
    return v,e

#implementing dijkstra
def dijkstra_raw(src,dst):
    global count
    global vertices
    global edges
    ver,edg=regular(vertices,edges)
#    admat = adjacencyMat(ver,edg)
    edg = edg.tolist()
    #q=[admat[src][i] for i in range(0,count)]
    g = defaultdict(list)
    for l,r,c in edg:
        g[l].append((c,r))

    q, seen = [(0,src,())], set()
    while q:
		 (cost,v1,path) = heapq.heappop(q)
		 if v1 not in seen:
			 seen.add(v1)
			 path = (v1, path)
			 if v1 == dst:
				 return cost,path
			 for c, v2 in g.get(v1, ()):
				 if v2 not in seen:
					 heapq.heappush(q, (cost+c, v2, path))


    return float("inf"),[]

def dijkstra(src,dst):
    len_shortest_path = -1
    ret_path=[]
    length,path_queue = dijkstra_raw(src,dst)
    if len(path_queue)>0:
        len_shortest_path = length		## 1. Get the length firstly;
        		## 2. Decompose the path_queue, to get the passing nodes in the shortest path.
        left = path_queue[0]
        ret_path.append(left)		## 2.1 Record the destination node firstly;
        right = path_queue[1]
        while len(right)>0:
            left = right[0]
            ret_path.append(left)	## 2.2 Record other nodes, till the source-node.
            right = right[1]
        ret_path.reverse()	## 3. Reverse the list finally, to make it be normal sequence.
    return len_shortest_path,ret_path



def astar(src,dst):
    global count
    global vertices
    global edges
    ver,edg=regular(vertices,edges)
    s_x = ver[src][1]
    s_y = ver[src][2]
    e_x = ver[dst][1]
    e_y = ver[dst][2]
    result = A_star(s_x,s_y,e_x,e_y)
    path = result.cal_path()
    return path

class A_star:
    def __init__(self, s_x, s_y, e_x, e_y):
        self.s_x = s_x
        self.s_y = s_y
        self.e_x = e_x
        self.e_y = e_y
        self.opened = {}
        self.closed = set()
        self.path = []
        self.dead = set()
    # find path
    def cal_path(self):
        p = Node(None, self.s_x, self.s_y, 0.0)
        self.opened[p.id] = p
        # travel open set until it is empty
        while len(self.opened) > 0:
            # node with best f
            idx, node = self.get_best()
            # return path if node is the dst
            if node.x == self.e_x and node.y == self.e_y:
                self.make_path(node)
                return
            # put node with best f in close, delete it from open
            self.closed.add(node.id)
            del self.opened[node.id]
            # get point around node
            self.get_neighboor(node)
    # return path to dst
    def make_path(self, p):
        while p:
            self.path.append((p.x, p.y))
            p = p.parent
        self.path.reverse()
        # delete src
        if len(self.path) > 1:
            del self.path[0]
    # travel opened to get best f 
    def get_best(self):
        best = None
        f_cost = 10000000000
        bi = None
        for key, item in self.opened.iteritems():
            value = self.get_f(item)
            if value < f_cost:
                best = item
                f_cost = value
                bi = key
        return bi, best
    # get f
    def get_f(self, i):
        return i.g_cost + (self.e_x - i.x)**2 + (self.e_y - i.y)**2
    # get node around p
    def get_neighboor(self, p):
        # 4 directions
        xs = (0, -1, 1, 0)
        ys = (-1, 0, 0, 1)
        for x, y in zip(xs, ys):
            new_x, new_y = x + p.x, y + p.y
            # if node is reachable
            if self.is_valid(new_x, new_y) is False:
                continue
            # new node
            node = Node(p, new_x, new_y, p.g_cost+self.get_cost(p.x, p.y, new_x, new_y))
            if node.id in self.closed:
                continue
            # node is in open set
            if node.id in self.opened:
                #g in open is bigger than node's,so g for node is g in open
                if self.opened[node.id].g_cost > node.g_cost:
                    self.opened[node.id].parent = p
                    self.opened[node.id].g_cost = node.g_cost
                    # self.f_cost[node.id] = self.get_f(node)
                else:
                    # add node to open set if its not in there
                    self.opened[node.id] = node
    # if node is reachable
    def is_valid(self, x, y):
        if abs(x) > 150 or y < -80 or y > 300:
            return False
        tmp_str = str(x) + '|' + str(y)
        
        if tmp_str in self.dead:
            return False
        # travel wall
        for index, value in enumerate(conf.WALLS[1]):
            if abs(x - value[0]) <= 25 and abs(y - value[1]+25) <= 5:
                self.dead.add(tmp_str)
                return False
        return True
    # get g
    def get_cost(self, x1, y1, x2, y2):
        if x1 == x2 or y1 == y2:
            return 1.0
        return 1.4
# clasas node
class Node(object):
    def __init__(self, parent, x, y, g_cost):
        self.parent = parent
        self.x = x
        self.y = y
        self.id = str(x) + '|' + str(y)
        self.g_cost = g_cost






#__main__function      
if __name__ == '__main__':
#    filepath = raw_input("Please input the path of graph:")
    readfile('F:/project-gwu/Artificial Intelligence/graph100_520.txt')
#    print 'vertices is:',vertices
#    print 'edges is:',edges
    count = len(vertices)
    countE = len(edges)
#    print count,countE
##    print len(vertices)
##    print len(edges)
##    print edges[0][1],vertices[0][2]
#    ver,edg=regular(vertices,edges)
#    print ver
#    print edg
#    admat=adjacencyMat(ver,edg)
#    print admat

    length,shortest = dijkstra(20,56)
    
    print "The shortest path is ",shortest,",it cost ",length
    
    
    path = astar(0,56)
    print "The path for A* is",path

    