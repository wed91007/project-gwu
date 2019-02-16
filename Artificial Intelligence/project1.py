import numpy as np
import heapq
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
def dijkstra(src,dst):
    global count
    global vertices
    global edges
    ver,edg=regular(vertices,edges)
    admat = adjacencyMat(ver,edg)
    
    #q=[admat[src][i] for i in range(0,count)]

    q = np.array(admat[src])
    seen = set()
    cost=0
#    print type(q)
    
#    v1 = np.where(cost)
#    print cost
#    print type(v1)
#    while q:
#    cost = heapq.heappop(q)
#    print type(cost)

    while q: 
        
        cost = q.min()
        v1 =  int(np.argwhere(q==cost))
        q[v1]=np.inf
        if v1 not in seen:
            seen.add(v1)
            path = (v1,path)
            if v1 == dst:return(cost,path)

            for v2 in range(0,count):
                if admat[v1][v2] != np.inf:
                    if v2 in seen:continue
                    prev = np.array(admat[v2]).min()
                    next = cost + admat[v1][v2]
                    if prev is inf or next < prev:

    return np.inf   

#        if v1 not in seen:
#			seen.add(v1)
#			path = (v1, path)
#			if v1 == dst:
#				return cost,path
#			for c, v2 in g.get(v1, ()):
#				if v2 not in seen:
#					heapq.heappush(q, (cost+c, v2, path))
#	return float("inf"),[]


    
#__main__function      
if __name__ == '__main__':
#    filepath = raw_input("Please input the path of graph:")
    readfile('/Users/hanxiangyang/git/project-gwu/Artificial Intelligence/graph100_520.txt')
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
    dijkstra(0,2)
    print np.inf