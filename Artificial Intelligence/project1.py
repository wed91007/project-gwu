import numpy as np



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
    admat = np.zeros((count,count),dtype=np.int)
    for i in range(0,countE):
        x=int(edges[i][0])
        y=int(edges[i][1])
        cost=int(edges[i][2])
        admat[y][x]=admat[x][y] = cost
    return admat
        
        
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


        
#__main__function      
if __name__ == '__main__':
#    filepath = raw_input("Please input the path of graph:")
    readfile('F:/Python/ai/data/graphs/graph100_520.txt')
    print 'vertices is:',vertices
    print 'edges is:',edges
    count = len(vertices)
    countE = len(edges)
    print count,countE
#    print len(vertices)
#    print len(edges)
#    print edges[0][1],vertices[0][2]
    ver,edg=regular(vertices,edges)
    print ver
    print edg
    admat=adjacencyMat(ver,edg)
    print admat