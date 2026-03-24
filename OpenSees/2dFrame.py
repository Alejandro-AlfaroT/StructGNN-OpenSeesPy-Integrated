import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt

ops.wipe()
ops.model('basic','-ndm', 2, '-ndf', 3)

colL, girL= 6.,8.

Acol, Agir = 2.e-3, 6.e-3
IzCol, IzGir = 1.6e-5, 5.4e-5

E= 200.e9

Ep = {1: [E, Acol, IzCol],
      2: [E, Acol, IzCol],
      3: [E, Agir, IzGir]}
for node_index in range(1,11):
    x_cord=0.
    y_cord=node_index*colL
    ops.node(node_index,x_cord,y_cord)
for node_index in range(11,21):
    x_cord=girL
    y_cord=(node_index-10)*colL
    ops.node(node_index,x_cord,y_cord)
for node_index in range(21,31):
    x_cord=girL*2
    y_cord=(node_index-20)*colL
    ops.node(node_index,x_cord,y_cord)
for node_index in range(31,41):
    x_cord=girL*3
    y_cord=(node_index-30)*colL
    ops.node(node_index,x_cord,y_cord)
for node_index in range(41,51):
    x_cord=girL*4
    y_cord=(node_index-40)*colL
    ops.node(node_index,x_cord,y_cord)

ops.fix(1,1,1,1)
ops.fix(11,1,1,1)
ops.fix(21,1,1,1)
ops.fix(31,1,1,1)
ops.fix(41,1,1,1)


opsv.plot_model()
plt.title('plot_model before defining elements')


ops.geomTransf('Linear',1)

#Column Definition
for a in range(1,10):
    ops.element('elasticBeamColumn',a,a,a+1,Acol,E,IzCol,1)
for a in range(10,19):
    ops.element('elasticBeamColumn',a,a+1,a+2,Acol,E,IzCol,1)
for a in range(19,28):
    ops.element('elasticBeamColumn',a,a+2,a+3,Acol,E,IzCol,1)
for a in range(28,37):
    ops.element('elasticBeamColumn',a,a+3,a+4,Acol,E,IzCol,1)
for a in range(37,46):
    ops.element('elasticBeamColumn',a,a+4,a+5,Acol,E,IzCol,1)
#Beam Definition
for a in range(46,50):
    ops.element('elasticBeamColumn',a,2+(a-46)*10,(a-46)*10+12,Agir,E,IzGir,1)
for a in range(50,54):
    ops.element('elasticBeamColumn',a,3+(a-50)*10,(a-50)*10+13,Agir,E,IzGir,1)
for a in range(54,58):
    ops.element('elasticBeamColumn',a,4+(a-54)*10,(a-54)*10+14,Agir,E,IzGir,1)
for a in range(58,62):
    ops.element('elasticBeamColumn',a,5+(a-58)*10,15+(a-58)*10,Agir,E,IzGir,1)
for a in range(62,66):
    ops.element('elasticBeamColumn',a,6+(a-62)*10,16+(a-62)*10,Agir,E,IzGir,1)
for a in range(66,70):
    ops.element('elasticBeamColumn',a,7+(a-66)*10,17+(a-66)*10,Agir,E,IzGir,1)
for a in range(70,74):
    ops.element('elasticBeamColumn', a, 8+(a-70)*10, 18+(a-70)*10,Agir,E,IzGir,1)
for a in range(74,78):
    ops.element('elasticBeamColumn', a, 9+(a-74)*10, 19+(a-74)*10,Agir,E,IzGir,1)
for a in range(78,82):
    ops.element('elasticBeamColumn', a,10+(a-78)*10,20+(a-78)*10,Agir,E,IzGir,1)

Px = 4.e+3
Py = -3.e+3
M = -150
Wy = -10.e+3
Wx = 0.

Ew= {3: ['beamUniform',Wy,Wx],
     6: ['beamUniform',Wy,Wx]}

ops.timeSeries('Constant',1)
ops.pattern('Plain',1,1)
ops.load(2,Px,0.,0.)
ops.load(5,Px,0.,0.)
ops.load(6,0.,Py,0.)
ops.load(1,0.,0.,M)

for etag in Ew:
    ops.eleLoad('-ele', etag, '-type', Ew[etag][0],Ew[etag][1],
                Ew[etag][2])
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr',1.0e-6, 6, 2)
ops.algorithm('Linear')
ops.integrator('LoadControl',1)
ops.analysis('Static')
ops.analyze(1)

ops.printModel()

opsv.plot_model()
plt.title('plot_model after defining elements')

opsv.plot_load()
plt.title("Load Scenario")

opsv.plot_defo()
plt.title("Structure Deformation")

sfacN,sfacV,sfacM = 5.e-5, 5.e-5, 5.e-5

opsv.section_force_diagram_2d('N',sfacN)
plt.title("Axial force Distribution")

opsv.section_force_diagram_2d('T',sfacV)
plt.title("Shear force Distribution")

opsv.section_force_diagram_2d('M',sfacM)
plt.title("Bending Moment Distribution")

plt.show()

exit()