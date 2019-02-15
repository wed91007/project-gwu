

#read data
f = open('F:/Python/ai/data/graphs/graph100_520.txt','r')
result = list()
vertices=list()
edges=list()


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
#print result
f = open('F:/Python/ai/data/graphs/graph100_520.txt','r')
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
print 'vertices is:',vertices
print 'edges is:',edges